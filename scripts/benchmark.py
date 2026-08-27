"""
Run the QASPER benchmark and the retrieval ablation, and write a JSON report.

The ablation adds one component at a time, so each row's difference from the
row above is attributable to that component alone:

    vector          vector search only
    hybrid          + BM25, fused with RRF
    hybrid_rerank   + cross-encoder reranking
    full            + query processing (section routing, academic filters)

Usage:
    python scripts/benchmark.py --papers 40
    python scripts/benchmark.py --papers 40 --answers 25       # + answer quality
    python scripts/benchmark.py --papers 5 --arms vector,hybrid --verbose
    python scripts/benchmark.py --reuse-corpus                 # skip re-ingestion

The corpus is ingested under a dedicated user id, so it is isolated from any
real user's documents and can be rebuilt without touching them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make `docusense` importable when run as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402

DEFAULT_DATASET = "data/benchmarks/qasper/qasper-test-v0.3.json"
DEFAULT_OUTPUT = "data/benchmarks/qasper_ablation_report.json"

# Metrics shown in the console table, in order.
TABLE_METRICS = ["MRR", "NDCG@10", "P@1", "Recall@5", "Recall@10", "MAP"]


def configure_logging(verbose: bool) -> None:
    """Quiet the pipeline's own logging unless asked for it."""
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "WARNING")

    # A full run takes tens of minutes. Redirected to a file, block-buffered
    # stdout shows nothing at all until the process exits, which is
    # indistinguishable from a hang.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):  # not a regular stream
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the QASPER retrieval benchmark and ablation."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help=f"QASPER JSON file (default: {DEFAULT_DATASET})")
    parser.add_argument("--papers", type=int, default=40,
                        help="Number of papers to ingest (default: 40)")
    parser.add_argument("--questions", type=int, default=None,
                        help="Cap on measurable questions (default: all)")
    parser.add_argument("--seed", type=int, default=20260827,
                        help="Seed for paper sampling (default: 20260827)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Results retrieved per query (default: 10)")
    parser.add_argument("--arms", default=None,
                        help="Comma-separated arm names (default: the four-arm ladder; 'legacy_default' is also available)")
    parser.add_argument("--answers", type=int, default=0,
                        help="Also score answer quality on N questions (needs Ollama)")
    parser.add_argument("--reuse-corpus", action="store_true",
                        help="Do not re-ingest; reuse the corpus already stored")
    parser.add_argument("--only-routed", action="store_true",
                        help="Keep only questions that section routing fires on, "
                             "to measure routing where it actually applies")
    parser.add_argument("--out", default=DEFAULT_OUTPUT,
                        help=f"Report path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--verbose", action="store_true",
                        help="Show pipeline logs")
    return parser.parse_args()


def print_table(arm_results) -> None:
    """Print the ablation as a fixed-width table with deltas vs the baseline."""
    print()
    print("Retrieval ablation")
    print("=" * 96)

    header = f"{'arm':<34}" + "".join(f"{m:>10}" for m in TABLE_METRICS) + f"{'ms/query':>12}"
    print(header)
    print("-" * 96)

    baseline = None
    for arm_result in arm_results:
        metrics = arm_result.result.retrieval
        if metrics is None:
            print(f"{arm_result.arm.label:<34}  (no retrieval metrics)")
            continue

        values = metrics.to_dict()
        if baseline is None:
            baseline = values

        row = f"{arm_result.arm.label:<34}"
        row += "".join(f"{values[m]:>10.4f}" for m in TABLE_METRICS)
        row += f"{arm_result.mean_latency_ms:>12.0f}"
        print(row)

    if baseline and len(arm_results) > 1:
        print("-" * 96)
        print("Change vs vector-only baseline:")
        for arm_result in arm_results[1:]:
            metrics = arm_result.result.retrieval
            if metrics is None:
                continue
            values = metrics.to_dict()
            deltas = []
            for m in TABLE_METRICS:
                base = baseline[m]
                delta = values[m] - base
                pct = (delta / base * 100) if base else float("nan")
                deltas.append(f"{m} {delta:+.4f} ({pct:+.1f}%)")
            print(f"  {arm_result.arm.label}: " + ", ".join(deltas))
    print("=" * 96)
    print()


def print_comparisons(comparisons) -> None:
    """Print paired bootstrap comparisons between successive arms."""
    if not comparisons:
        return

    print("Paired comparisons (10,000 bootstrap resamples over the same queries)")
    print("=" * 96)
    for c in comparisons:
        verdict = "significant" if c.significant else "NOT significant"
        print(
            f"  {c.metric:<8} {c.baseline:>16} -> {c.variant:<16} "
            f"{c.delta:+.4f}  95% CI [{c.ci_low:+.4f}, {c.ci_high:+.4f}]  "
            f"p={c.p_value:.4f}  {verdict}"
        )
    print("=" * 96)
    print()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    from docusense.evaluation.qasper_harness import (
        ALL_ARMS,
        DEFAULT_ARMS,
        QASPERHarness,
        paired_bootstrap,
    )

    dataset = Path(args.dataset)
    if not dataset.exists():
        print(f"ERROR: dataset not found: {dataset}", file=sys.stderr)
        print(
            "Download it with:\n"
            "  curl -O https://qasper-dataset.s3.us-west-2.amazonaws.com/"
            "qasper-test-and-evaluator-v0.3.tgz\n"
            f"  tar xzf qasper-test-and-evaluator-v0.3.tgz -C {dataset.parent}",
            file=sys.stderr,
        )
        return 1

    arms = DEFAULT_ARMS
    if args.arms:
        wanted = {name.strip() for name in args.arms.split(",") if name.strip()}
        arms = [a for a in ALL_ARMS if a.name in wanted]
        unknown = wanted - {a.name for a in ALL_ARMS}
        if unknown:
            print(f"ERROR: unknown arm(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            print(f"Available: {', '.join(a.name for a in ALL_ARMS)}", file=sys.stderr)
            return 1
        if not arms:
            print("ERROR: no arms selected", file=sys.stderr)
            return 1

    started = time.time()
    harness = QASPERHarness(dataset_path=dataset)

    print(f"Preparing corpus: {args.papers} papers (seed {args.seed})...")
    samples, stats = harness.prepare(
        num_papers=args.papers,
        seed=args.seed,
        max_questions=args.questions,
        reset=not args.reuse_corpus,
    )

    print(json.dumps(stats.to_dict(), indent=2))

    if not samples:
        print(
            "ERROR: no measurable questions. The corpus is empty or no evidence "
            "could be grounded to a chunk.",
            file=sys.stderr,
        )
        return 1

    routed_only = False
    if args.only_routed:
        # Averaged over every question, section routing is diluted by the
        # majority it never fires on. This isolates the questions it does act
        # on, where a gain or loss is actually attributable to it.
        from docusense.retrieval.query_processor import QueryProcessor

        processor = QueryProcessor()
        before = len(samples)
        samples = [
            s for s in samples if processor.detect_section_intent(s.query)
        ]
        routed_only = True
        print(f"Section routing fires on {len(samples)}/{before} questions")
        if not samples:
            print("ERROR: section routing fires on no question", file=sys.stderr)
            return 1

    arm_results = []
    for arm in arms:
        print(f"Running arm: {arm.label} ...")
        arm_result = harness.run_arm(arm, samples, top_k=args.top_k)
        arm_results.append(arm_result)

    print_table(arm_results)

    # Compare each arm against the one before it, so a difference is
    # attributable to the single component that changed, plus every arm against
    # the vector-only baseline.
    pairs = [(a.arm.name, b.arm.name) for a, b in zip(arm_results, arm_results[1:])]
    pairs += [(arm_results[0].arm.name, o.arm.name) for o in arm_results[1:]]

    by_name = {r.arm.name: r for r in arm_results}
    comparisons = []
    seen_pairs = set()
    for baseline_name, variant_name in pairs:
        if baseline_name == variant_name or (baseline_name, variant_name) in seen_pairs:
            continue
        seen_pairs.add((baseline_name, variant_name))
        comparisons.append(paired_bootstrap(
            by_name[baseline_name].per_query_rr,
            by_name[variant_name].per_query_rr,
            "MRR", baseline_name, variant_name,
        ))

    print_comparisons(comparisons)

    report = {
        "benchmark": "qasper_retrieval_ablation",
        "dataset": str(dataset),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "papers": args.papers,
            "seed": args.seed,
            "top_k": args.top_k,
            "question_cap": args.questions,
            "only_routed": routed_only,
        },
        "questions_measured": len(samples),
        "corpus": stats.to_dict(),
        "arms": [r.to_dict() for r in arm_results],
        "paired_comparisons": [c.to_dict() for c in comparisons],
    }

    if args.answers:
        print(f"Scoring answer quality on {args.answers} questions (Ollama)...")
        answer_result, _ = harness.evaluate_answers(samples, limit=args.answers)
        metrics = answer_result.answer.to_dict() if answer_result.answer else {}

        # QASPER carries no author or venue metadata, and the reconstructed
        # documents therefore have none either -- what the extractor recovers
        # as "authors" is fragments of the title. Citation accuracy is scored
        # by matching cited surnames against source metadata, so on this corpus
        # it measures the corpus, not the system. Reporting the resulting 0.0
        # as a product number would be a false claim, so the figures are moved
        # aside and labelled.
        not_applicable = {
            key: metrics.pop(key)
            for key in ("Citation-P", "Citation-R", "Citation-F1", "has_citations")
            if key in metrics
        }

        report["answer_quality"] = {
            "num_samples": answer_result.num_samples,
            "note": (
                "End-to-end rag.ask() on a deterministic subsample, with "
                "llama3.2:3b. ROUGE compares prose answers against QASPER's "
                "short extractive reference spans, so it reads low by "
                "construction; treat it as a regression signal, not an "
                "accuracy score."
            ),
            "metrics": metrics,
            "not_applicable": {
                "reason": (
                    "QASPER has no author/venue metadata, so citation accuracy "
                    "cannot be measured against it."
                ),
                "values": not_applicable,
            },
        }
        if answer_result.answer:
            print()
            print("Answer quality")
            print("-" * 60)
            for key, value in answer_result.answer.to_dict().items():
                print(f"  {key:<24} {value}")
            print()

    report["total_seconds"] = round(time.time() - started, 1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Report written to {out}  ({report['total_seconds']}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
