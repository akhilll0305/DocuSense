"""
End-to-end QASPER: retrieve, answer, cite evidence -- scored with the official
Answer-F1 and Evidence-F1.

    python scripts/qasper_answers.py --papers 80                 # Ollama
    python scripts/qasper_answers.py --provider groq --limit 100
    python scripts/qasper_answers.py --scope corpus              # harder setting

THE TASK SETTING
----------------
QASPER asks a question *about a given paper*: the published baselines read
that paper and nothing else. `--scope paper` (the default) reproduces that --
retrieval is filtered to the question's own document -- so the numbers are
comparable with the paper's. `--scope corpus` searches the whole 80-paper shelf
instead, which is what the retrieval ablation in BENCHMARKS.md measures and
is strictly harder: the system has to find the paper before the paragraph.

Every question in the sampled papers is asked, unanswerable ones included,
because the official metric scores those too ("Unanswerable" is a reference
answer, and saying it when it is right is worth a full point).

WHAT IS SUBMITTED AS THE ANSWER
-------------------------------
Not the product's prose. QASPER's references are short -- a span, a list,
Yes/No -- and token F1 against a 12-word span punishes a correct 150-word
paragraph for every word it adds. So the same retrieval and the same model
are asked for the answer in QASPER's own form. That is a different prompt,
and it is labelled as one in the report; the retrieval underneath is the
shipped pipeline.

EVIDENCE
--------
QASPER evidence is paragraphs; retrieval returns ~500-token chunks. The
paragraphs of the paper that sit inside the top retrieved chunks are scored
against the question by the same cross-encoder the reranker uses, and the
best `--evidence-k` are submitted. An "Unanswerable" answer submits none,
which is what the metric expects of one.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402

DATASET = "data/benchmarks/qasper/qasper-test-v0.3.json"
OFFICIAL_SCRIPT = "data/benchmarks/qasper/qasper_evaluator.py"

SYSTEM_PROMPT = (
    "You answer questions about one scientific paper, using only the excerpts "
    "given. Reply with the shortest answer that is complete:\n"
    "- a short phrase, number, or list, copied from the excerpts where possible;\n"
    "- 'Yes' or 'No' for a yes/no question;\n"
    "- 'Unanswerable' if the excerpts do not contain the answer.\n"
    "No explanation, no citation, no full sentence."
)

_THINK = re.compile(r"<think>.*?</think>", re.S)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QASPER Answer-F1 / Evidence-F1")
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--papers", type=int, default=80)
    p.add_argument("--seed", type=int, default=20260827)
    p.add_argument("--scope", choices=["paper", "corpus"], default="paper")
    p.add_argument("--top-k", type=int, default=5, help="Chunks given to the model")
    p.add_argument("--evidence-k", type=int, default=2, help="Paragraphs submitted as evidence")
    p.add_argument("--provider", default=None, help="ollama | groq (default: LLM_PROVIDER)")
    p.add_argument("--model", default=None, help="Override the provider's model")
    p.add_argument("--limit", type=int, default=None, help="Cap on questions (random, seeded)")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--out", default=None)
    return p.parse_args()


def clean_answer(text: str) -> str:
    """Strip reasoning blocks and wrapping a model adds despite being asked not to."""
    text = _THINK.sub("", text or "").strip()
    text = text.strip().strip('"').strip()
    first = text.splitlines()[0].strip() if text else ""
    # "Answer: X" -> "X"
    first = re.sub(r"^(final\s+)?answer\s*:\s*", "", first, flags=re.I)
    if re.match(r"^unanswerable\b", first, re.I):
        return "Unanswerable"
    return first


def build_client(provider, model):
    from docusense.config.settings import settings

    name = (provider or settings.llm_provider or "ollama").lower()
    if name == "ollama":
        from docusense.llms.ollama_client import OllamaClient
        return OllamaClient(model=model, temperature=0.0, max_tokens=96)
    if name == "groq":
        from docusense.llms.groq_client import GroqClient
        return GroqClient(model=model, temperature=0.0, max_tokens=256)
    raise SystemExit(f"unknown provider {name}")


def main() -> int:
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    from docusense.embeddings.backends import load_cross_encoder_backend
    from docusense.config.settings import settings
    from docusense.evaluation.qasper_harness import (
        DEFAULT_ARMS,
        QASPERHarness,
        normalize_text,
    )
    from docusense.evaluation.qasper_official import evaluate, get_answers_and_evidence

    dataset = Path(args.dataset)
    raw = json.loads(dataset.read_text(encoding="utf-8"))

    harness = QASPERHarness(dataset_path=dataset)
    papers = harness.select_papers(args.papers, seed=args.seed)
    harness.load_existing_document_ids()
    missing = [p for p in papers if p.paper_id not in harness.document_ids]
    if missing:
        print(f"Ingesting {len(missing)} paper(s) not yet in the corpus...")
        harness.ingest_papers(missing, skip_existing=True)

    arm = next(a for a in DEFAULT_ARMS if a.name == ("hybrid" if args.no_rerank else "hybrid_rerank"))
    pipeline = harness._build_pipeline(arm)
    cross_encoder = load_cross_encoder_backend(settings.reranker_model)
    client = build_client(args.provider, args.model)
    if not client.is_available():
        print(f"ERROR: LLM backend not available ({client.model})", file=sys.stderr)
        return 1

    questions = [(paper, q) for paper in papers for q in paper.questions]
    if args.limit and args.limit < len(questions):
        picked = sorted(random.Random(args.seed).sample(range(len(questions)), args.limit))
        questions = [questions[i] for i in picked]

    print(f"{len(questions)} questions over {len(papers)} papers, scope={args.scope}, "
          f"model={client.model}")

    predictions = {}
    latencies = []
    started = time.time()

    for i, (paper, question) in enumerate(questions, 1):
        t0 = time.time()
        filters = {"user_id": harness.user_id}
        if args.scope == "paper":
            filters["document_id"] = harness.document_ids[paper.paper_id]

        try:
            results, _ = pipeline.retrieve(question.question, top_k=args.top_k, filters=filters)
        except Exception as exc:
            logger.warning(f"retrieval failed: {exc}")
            results = []

        context = "\n\n".join(
            f"[Excerpt {n}]\n{r.text}" for n, r in enumerate(results, 1)
        )
        prompt = f"Excerpts:\n\n{context}\n\nQuestion: {question.question}\nAnswer:"
        try:
            answer = clean_answer(client.generate(prompt, system_prompt=SYSTEM_PROMPT))
        except Exception as exc:
            logger.warning(f"generation failed: {exc}")
            answer = ""

        evidence = []
        if answer and answer != "Unanswerable":
            # Paragraphs of the question's own paper found inside the top
            # chunks; ranked by the cross-encoder.
            chunk_texts = [normalize_text(r.text) for r in results[:3]]
            candidates = []
            for para in paper.paragraphs():
                norm = normalize_text(para)
                if len(norm) > 20 and any(norm in c for c in chunk_texts):
                    candidates.append(para)
            if candidates:
                scores = cross_encoder.predict([(question.question, c) for c in candidates])
                ranked = sorted(zip(scores, candidates), key=lambda x: -x[0])
                evidence = [c for _, c in ranked[: args.evidence_k]]

        predictions[question.question_id] = {"answer": answer, "evidence": evidence}
        latencies.append(time.time() - t0)

        if i % 20 == 0 or i == len(questions):
            print(f"  {i}/{len(questions)}  ({time.time() - started:.0f}s)")

    gold = get_answers_and_evidence(
        raw, text_evidence_only=True, paper_ids=[p.paper_id for p in papers]
    )
    if args.limit:
        gold = {qid: refs for qid, refs in gold.items() if qid in predictions}
    result = evaluate(gold, predictions)

    # Cross-check against AllenAI's own script on the same predictions.
    official = None
    if Path(OFFICIAL_SCRIPT).exists():
        with tempfile.TemporaryDirectory() as tmp:
            subset = {pid: raw[pid] for pid in (p.paper_id for p in papers)}
            if args.limit:
                subset = {
                    pid: {**data, "qas": [q for q in data["qas"] if q["question_id"] in predictions]}
                    for pid, data in subset.items()
                }
            gold_path = Path(tmp) / "gold.json"
            pred_path = Path(tmp) / "pred.jsonl"
            gold_path.write_text(json.dumps(subset), encoding="utf-8")
            pred_path.write_text("\n".join(
                json.dumps({"question_id": qid, "predicted_answer": p["answer"],
                            "predicted_evidence": p["evidence"]})
                for qid, p in predictions.items()
            ), encoding="utf-8")
            out = subprocess.run(
                [sys.executable, OFFICIAL_SCRIPT, "--predictions", str(pred_path),
                 "--gold", str(gold_path), "--text_evidence_only"],
                capture_output=True, text=True,
            )
            if out.returncode == 0:
                official = json.loads(out.stdout)
                for key in ("Answer F1", "Evidence F1"):
                    if abs(official[key] - result[key]) > 1e-9:
                        print(f"ERROR: {key} disagrees with the official script: "
                              f"{result[key]} vs {official[key]}", file=sys.stderr)
                        return 1
            else:
                print(f"official script failed: {out.stderr[-500:]}", file=sys.stderr)

    per_question = result.pop("per_question")
    report = {
        "benchmark": "qasper_answers",
        "dataset": str(dataset),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "papers": len(papers), "seed": args.seed, "scope": args.scope,
            "top_k": args.top_k, "evidence_k": args.evidence_k,
            "retrieval_arm": arm.name, "embedding_model": settings.embedding_model,
            "provider": type(client).__name__, "model": client.model,
            "question_cap": args.limit, "prompt": "concise (QASPER answer form)",
        },
        "questions": len(questions),
        "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items()},
        "verified_against_official_script": official is not None,
        "mean_seconds_per_question": round(sum(latencies) / max(len(latencies), 1), 2),
        "per_question": per_question,
        "predictions": predictions,
    }

    out = Path(args.out or f"data/benchmarks/qasper_answers_{args.scope}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    m = report["metrics"]
    print()
    print(f"Answer F1    {m['Answer F1']:.4f}")
    for t, v in m["Answer F1 by type"].items():
        print(f"  {t:<12}{v:.4f}  (n={m['Questions by type'][t]})")
    print(f"Evidence F1  {m['Evidence F1']:.4f}")
    print(f"official script agrees: {official is not None}")
    print(f"Report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
