"""
Fine-tune the cross-encoder reranker on QASPER's train split, export to ONNX.

    python scripts/finetune_reranker.py
    python scripts/finetune_reranker.py --epochs 1 --papers 50   # smoke run

The reranker is the component the ablation credits with the largest gain
(+25.8% MRR over hybrid search), and the stock one was trained on MS MARCO --
web queries over web passages. QASPER questions over paper chunks are a
different distribution, which is the case fine-tuning is for.

DATA
----
The embedder's training pairs, read as labelled (question, chunk) examples:
each evidence chunk is a 1, and each of the question's same-paper BM25 hard
negatives (ranks 2-4; see docusense/training/qasper_pairs.py) is a 0. Trained
with binary cross-entropy on the logit, which is what the base model's single
output already is.

SELECTION
---------
After each epoch the model reranks a realistic first stage on QASPER *dev*:
BM25's top 20 over every dev chunk pooled, positives not forced in. MRR@10 of
that ranking picks the checkpoint. The test split is never read here.

OUTPUT
------
    models/<name>/                  CrossEncoder checkpoint
    models/<name>/onnx/model.onnx   what MODEL_RUNTIME=onnx loads
    data/benchmarks/reranker_finetune_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402

TRAIN = "data/benchmarks/qasper/qasper-train-v0.3.json"
DEV = "data/benchmarks/qasper/qasper-dev-v0.3.json"
_TOKEN = re.compile(r"\w+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--name", default="reranker-qasper")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--papers", type=int, default=None)
    p.add_argument("--candidates", type=int, default=20, help="First-stage depth on dev")
    p.add_argument("--eval-steps", type=int, default=None,
                   help="Evaluate on dev every N steps and keep the best "
                        "checkpoint (early stopping), instead of once per epoch")
    p.add_argument("--no-pos-weight", action="store_true",
                   help="Unweighted BCE instead of weighting positives by the imbalance")
    p.add_argument("--seed", type=int, default=20260827)
    p.add_argument("--report", default="data/benchmarks/reranker_finetune_report.json")
    return p.parse_args()


def dev_samples(papers, by_paper, depth):
    """BM25 top-`depth` over the pooled dev corpus, per answerable question."""
    from rank_bm25 import BM25Okapi

    from docusense.training.qasper_pairs import retrieval_eval_set

    queries, corpus, relevant = retrieval_eval_set(papers, by_paper)
    ids = list(corpus)
    bm25 = BM25Okapi([_TOKEN.findall(corpus[i].lower()) for i in ids])

    samples = []
    for qid, query in queries.items():
        scores = bm25.get_scores(_TOKEN.findall(query.lower()))
        top = sorted(range(len(ids)), key=lambda i: -scores[i])[:depth]
        samples.append({
            "query": query,
            "positive": [corpus[c] for c in relevant[qid]],
            "documents": [corpus[ids[i]] for i in top],
        })
    return samples


def export_onnx(model, out_dir: Path, probe) -> float:
    """Write out_dir/onnx/model.onnx; return the worst |logit difference| vs torch."""
    import numpy as np
    import onnxruntime as ort
    import torch

    hf = model.model.eval().to("cpu")
    tokenizer = model.tokenizer
    (out_dir / "special_tokens_map.json").write_text(
        json.dumps(tokenizer.special_tokens_map, indent=2), encoding="utf-8"
    )
    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    path = onnx_dir / "model.onnx"

    names = ["input_ids", "attention_mask", "token_type_ids"]
    sample = tokenizer([("export probe", "a passage")], return_tensors="pt")

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).logits

    dynamic = {n: {0: "batch", 1: "sequence"} for n in names}
    dynamic["logits"] = {0: "batch"}
    with torch.no_grad():
        torch.onnx.export(
            Wrapper(hf), tuple(sample[n] for n in names), str(path),
            input_names=names, output_names=["logits"], dynamic_axes=dynamic,
            opset_version=17, dynamo=False,
        )

    encoded = tokenizer(probe, padding=True, truncation=True,
                        max_length=model.max_length, return_tensors="np")
    feed = {n: encoded[n].astype(np.int64) for n in names}
    onnx_logits = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, feed)[0][:, 0]
    torch_logits = model.predict(probe, activation_fn=torch.nn.Identity(), convert_to_numpy=True)
    return float(np.abs(onnx_logits - torch_logits).max())


def main() -> int:
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    import torch
    from datasets import Dataset
    from sentence_transformers.cross_encoder import (
        CrossEncoder,
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    from docusense.evaluation.qasper_loader import QASPERLoader
    from docusense.training.qasper_pairs import build_pairs

    started = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    loader = QASPERLoader()
    train_papers = loader.load(TRAIN)
    dev_papers = loader.load(DEV)
    if args.papers:
        train_papers = train_papers[: args.papers]

    print(f"Chunking {len(train_papers)} train papers...")
    pairs, _, train_stats = build_pairs(train_papers, seed=args.seed)

    queries, passages, labels = [], [], []
    seen_negatives = set()
    for p in pairs:
        queries.append(p.query)
        passages.append(p.positive)
        labels.append(1.0)
        for negative in p.negatives:
            # A question with two evidence chunks yields two pairs sharing the
            # same negatives; count each (question, negative) once.
            key = (p.query, negative)
            if key in seen_negatives:
                continue
            seen_negatives.add(key)
            queries.append(p.query)
            passages.append(negative)
            labels.append(0.0)
    num_pos = int(sum(labels))
    num_neg = len(labels) - num_pos
    print(f"train: {num_pos} positives, {num_neg} negatives")
    train_dataset = Dataset.from_dict(
        {"query": queries, "passage": passages, "label": labels}
    ).shuffle(seed=args.seed)

    print(f"Chunking {len(dev_papers)} dev papers...")
    _, dev_chunks, _ = build_pairs(dev_papers, with_negatives=False)
    samples = dev_samples(dev_papers, dev_chunks, args.candidates)
    print(f"dev: {len(samples)} queries, BM25 top-{args.candidates} as candidates")

    model = CrossEncoder(args.base, max_length=args.max_length, device=device)
    evaluator = CrossEncoderRerankingEvaluator(
        samples=samples, at_k=10, always_rerank_positives=False,
        name="qasper-dev", batch_size=64,
    )

    print("Scoring the base reranker on dev...")
    base_scores = evaluator(model)

    out_dir = Path("models") / args.name
    training_args = CrossEncoderTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=device == "cuda",
        eval_strategy="steps" if args.eval_steps else "epoch",
        save_strategy="steps" if args.eval_steps else "epoch",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_qasper-dev_mrr@10",
        greater_is_better=True,
        logging_steps=50,
        seed=args.seed,
        report_to="none",
    )
    # Positives are outnumbered roughly 3:1, so they are weighted up to keep
    # the loss from being won by predicting 'not relevant' everywhere.
    pos_weight = None if args.no_pos_weight else torch.tensor(num_neg / max(num_pos, 1))
    loss = BinaryCrossEntropyLoss(model, pos_weight=pos_weight)
    trainer = CrossEncoderTrainer(
        model=model, args=training_args, train_dataset=train_dataset,
        loss=loss, evaluator=evaluator,
    )
    trainer.train()

    print("Scoring the fine-tuned reranker on dev...")
    tuned_scores = evaluator(model)
    model.save(str(out_dir))

    print("Exporting to ONNX...")
    probe = [(s["query"], d) for s in samples[:4] for d in s["documents"][:4]]
    max_diff = export_onnx(model, out_dir, probe)
    print(f"torch vs ONNX, max |logit difference| over {len(probe)} pairs: {max_diff:.6f}")
    if max_diff > 1e-3:
        print("ERROR: the ONNX export does not reproduce the torch model", file=sys.stderr)
        return 1

    key = "qasper-dev_"
    picked = ["mrr@10", "ndcg@10", "map", "base_mrr@10", "base_ndcg@10"]
    report = {
        "base_model": args.base,
        "output": str(out_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "eval_steps": args.eval_steps,
        "pos_weight": None if pos_weight is None else round(float(pos_weight), 3),
        "train": {**train_stats, "positives": num_pos, "negatives": num_neg},
        "dev": {
            "queries": len(samples),
            "first_stage": f"BM25 top-{args.candidates}, pooled dev corpus",
            "base": {m: round(base_scores[key + m], 4) for m in picked if key + m in base_scores},
            "fine_tuned": {m: round(tuned_scores[key + m], 4) for m in picked if key + m in tuned_scores},
        },
        "onnx_max_logit_diff_vs_torch": round(max_diff, 6),
        "device": device,
        "seconds": round(time.time() - started, 1),
    }
    # load_best_model_at_end picks the best *checkpoint*; the untouched base
    # model is not one of them. Say plainly when training never beat it.
    report["beats_base_on_dev"] = (
        report["dev"]["fine_tuned"].get("mrr@10", 0) > report["dev"]["base"].get("mrr@10", 0)
    )
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["dev"], indent=2))
    if not report["beats_base_on_dev"]:
        print("NOTE: the fine-tuned reranker does not beat the base model on dev.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
