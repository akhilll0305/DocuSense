"""
Fine-tune the embedding model on QASPER's train split, and export it to ONNX.

    python scripts/finetune_embedder.py                      # MiniLM, 3 epochs
    python scripts/finetune_embedder.py --epochs 1 --papers 50   # smoke run

Needs the torch runtime and, realistically, a GPU:

    pip install -r requirements-torch.txt accelerate datasets

WHAT IS MEASURED HERE, AND WHAT IS NOT
--------------------------------------
Model selection uses the *dev* split: after each epoch the model is scored on
retrieval over every dev paper's chunks, pooled, and the best epoch is kept.
The test split is never read by this script. The number that goes in the
README comes afterwards, from `scripts/benchmark.py` on test, through the full
pipeline -- the dev score below is only how the checkpoint was chosen.

The loss is MultipleNegativesRankingLoss: each question is pulled toward its
evidence chunk and pushed away from every other chunk in the batch, plus one
same-paper hard negative (see docusense/training/qasper_pairs.py). Batches
never hold the same question twice, or a question's own positive would be
counted against it as a negative.

OUTPUT
------
    models/<name>/                  sentence-transformers checkpoint
    models/<name>/onnx/model.onnx   what MODEL_RUNTIME=onnx loads
    data/benchmarks/finetune_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402

TRAIN = "data/benchmarks/qasper/qasper-train-v0.3.json"
DEV = "data/benchmarks/qasper/qasper-dev-v0.3.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--name", default="minilm-qasper")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-seq-length", type=int, default=256,
                   help="Must match the runtime's truncation (256 for MiniLM)")
    p.add_argument("--papers", type=int, default=None,
                   help="Cap on train papers, for a smoke run")
    p.add_argument("--no-negatives", action="store_true",
                   help="In-batch negatives only (for the ablation)")
    p.add_argument("--seed", type=int, default=20260827)
    p.add_argument("--report", default="data/benchmarks/finetune_report.json")
    return p.parse_args()


def export_onnx(model, out_dir: Path, probe) -> float:
    """
    Write `out_dir/onnx/model.onnx` and return its worst cosine against torch.

    optimum would do this, but its release at the time of writing imports
    names transformers 5 removed, so the export is done with torch.onnx
    directly: the encoder with input_ids / attention_mask / token_type_ids in
    and last_hidden_state out, dynamic in batch and sequence length.
    """
    import numpy as np
    import onnxruntime as ort
    import torch
    from tokenizers import Tokenizer

    encoder = model[0].auto_model.eval()
    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    path = onnx_dir / "model.onnx"

    sample = model.tokenizer(["export probe"], return_tensors="pt")
    names = ["input_ids", "attention_mask", "token_type_ids"]
    dynamic = {n: {0: "batch", 1: "sequence"} for n in names}
    dynamic["last_hidden_state"] = {0: "batch", 1: "sequence"}

    class Wrapper(torch.nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.enc(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).last_hidden_state

    with torch.no_grad():
        torch.onnx.export(
            Wrapper(encoder),
            tuple(sample[n] for n in names),
            str(path),
            input_names=names,
            output_names=["last_hidden_state"],
            dynamic_axes=dynamic,
            opset_version=17,
            dynamo=False,
        )

    # Read back through onnxruntime with the saved tokenizer, pooled the way
    # fastembed pools, and compare against sentence-transformers.
    tokenizer = Tokenizer.from_file(str(out_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=model.max_seq_length)
    tokenizer.enable_padding()
    encoded = tokenizer.encode_batch(probe)
    feed = {
        "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
        "token_type_ids": np.array([e.type_ids for e in encoded], dtype=np.int64),
    }
    hidden = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, feed)[0]
    mask = feed["attention_mask"][..., None].astype(np.float32)
    pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1e-9)
    pooled /= np.linalg.norm(pooled, axis=1, keepdims=True)

    reference = model.encode(probe, normalize_embeddings=True)
    return float((pooled * reference).sum(axis=1).min())


def main() -> int:
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    import torch
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    from docusense.evaluation.qasper_loader import QASPERLoader
    from docusense.training.qasper_pairs import build_pairs, retrieval_eval_set

    started = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    loader = QASPERLoader()
    train_papers = loader.load(TRAIN)
    dev_papers = loader.load(DEV)
    if args.papers:
        train_papers = train_papers[: args.papers]

    print(f"Chunking {len(train_papers)} train papers...")
    pairs, _, train_stats = build_pairs(
        train_papers, seed=args.seed, with_negatives=not args.no_negatives
    )
    print(json.dumps(train_stats))

    print(f"Chunking {len(dev_papers)} dev papers...")
    _, dev_chunks, dev_stats = build_pairs(dev_papers, with_negatives=False)
    queries, corpus, relevant = retrieval_eval_set(dev_papers, dev_chunks)
    print(f"dev: {len(queries)} queries over {len(corpus)} chunks")

    columns = {
        "anchor": [p.query for p in pairs],
        "positive": [p.positive for p in pairs],
    }
    if not args.no_negatives:
        # A pair without a negative (a one-chunk paper) borrows its own
        # positive's neighbour rather than being dropped; with the loss this
        # is just one more in-batch negative.
        fallback = [p.positive for p in pairs[1:]] + [pairs[0].positive]
        columns["negative"] = [
            p.negative if p.negative is not None else fb
            for p, fb in zip(pairs, fallback)
        ]
    train_dataset = Dataset.from_dict(columns).shuffle(seed=args.seed)

    model = SentenceTransformer(args.base, device=device)
    model.max_seq_length = args.max_seq_length

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant,
        name="qasper-dev",
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 10],
        precision_recall_at_k=[10],
        map_at_k=[10],
        show_progress_bar=False,
        batch_size=128,
    )

    print("Scoring the base model on dev...")
    base_scores = evaluator(model)

    out_dir = Path("models") / args.name
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=device == "cuda",
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_qasper-dev_cosine_mrr@10",
        greater_is_better=True,
        logging_steps=20,
        seed=args.seed,
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=MultipleNegativesRankingLoss(model),
        evaluator=evaluator,
    )
    trainer.train()

    print("Scoring the fine-tuned model on dev...")
    tuned_scores = evaluator(model)

    model.save(str(out_dir))

    # ONNX export. The file holds the transformer only -- token embeddings
    # out -- because that is the contract fastembed loads a custom model
    # under: it does the mean pooling and normalization itself. The checked
    # agreement below is what makes the ONNX file the same model rather than
    # a similar one.
    print("Exporting to ONNX...")
    model.to("cpu")
    agreement = export_onnx(model, out_dir, [queries[q] for q in list(queries)[:16]]
                            + list(corpus.values())[:16])
    print(f"torch vs ONNX, minimum cosine over 32 texts: {agreement:.6f}")
    if agreement < 0.9999:
        print("ERROR: the ONNX export does not reproduce the torch model", file=sys.stderr)
        return 1

    key = "qasper-dev_cosine_"
    picked = ["mrr@10", "ndcg@10", "accuracy@1", "recall@10", "map@10"]
    report = {
        "base_model": args.base,
        "output": str(out_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_seq_length": args.max_seq_length,
        "hard_negatives": not args.no_negatives,
        "train": train_stats,
        "dev": {
            "papers": dev_stats["papers"],
            "queries": len(queries),
            "chunks": len(corpus),
            "base": {m: round(base_scores[key + m], 4) for m in picked},
            "fine_tuned": {m: round(tuned_scores[key + m], 4) for m in picked},
        },
        "onnx_min_cosine_vs_torch": round(agreement, 6),
        "device": device,
        "seconds": round(time.time() - started, 1),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"{'dev metric':<14}{'base':>10}{'tuned':>10}")
    for m in picked:
        print(f"{m:<14}{report['dev']['base'][m]:>10.4f}{report['dev']['fine_tuned'][m]:>10.4f}")
    print(f"\nModel: {out_dir}   Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
