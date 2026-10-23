"""
Fail the build when a model gets worse.

    python scripts/quality_gate.py                    # check against the baseline
    python scripts/quality_gate.py --update-baseline  # accept the current numbers
    python scripts/quality_gate.py --build-fixtures   # regenerate the fixtures

The unit tests prove the code runs. They cannot see a retriever that returns
the wrong paragraphs, or a claim checker that has started passing everything:
both still return well-formed results. Each was a real risk this project ran
into -- a tokenizer misconfiguration that broke ingestion silently, an NLI model
that flagged three supported claims in four -- and each was caught only because
someone ran a benchmark by hand.

This runs the two models that decide what a reader sees on a small fixed slice
of the same benchmarks, and compares against the numbers in
tests/quality/baseline.json:

    retrieval      10 QASPER test papers ingested from scratch into an
                   in-memory store; MRR and Recall@10 of hybrid search with and
                   without the cross-encoder
    claim checks   120 labelled claims from QASPER dev, built the way
                   validate_claim_verifier.py builds them; ROC-AUC, and how
                   many supported claims get flagged

A metric may move by the tolerance in the baseline file and no further. The
fixtures are fixed and the models run at temperature-free inference, so a
clean checkout reproduces the baseline to the third decimal; the tolerance is
there for floating-point drift across CPUs, not for noise.

A deliberate change that moves a number -- a new embedder, a new threshold --
updates the baseline in the same commit, with --update-baseline, so the change
and its measured cost are reviewed together.

The fixtures are a slice of QASPER (CC BY 4.0, Dasigi et al., 2021). They are
committed so the gate needs no download; --build-fixtures regenerates them from
the full dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIXTURES = ROOT / "tests" / "quality"
RETRIEVAL_FIXTURE = FIXTURES / "qasper_slice.json"
CLAIMS_FIXTURE = FIXTURES / "claims.json"
BASELINE = FIXTURES / "baseline.json"

TEST_SET = ROOT / "data/benchmarks/qasper/qasper-test-v0.3.json"
DEV_SET = ROOT / "data/benchmarks/qasper/qasper-dev-v0.3.json"
SEED = 20260827

# Metrics where lower is better; every other one must not fall.
LOWER_IS_BETTER = {"claims.supported_flagged"}
DEFAULT_TOLERANCE = 0.02


def build_fixtures(papers: int, claims_per_label: int) -> None:
    """Cut the committed fixtures out of the full QASPER files."""
    test = json.loads(TEST_SET.read_text(encoding="utf-8"))

    # Papers drawn from those the full benchmark can measure, by the same
    # rule: at least one answerable question with body-paragraph evidence.
    from docusense.evaluation.qasper_harness import QASPERHarness

    # select_papers only reads the dataset; the store the constructor opens
    # is not needed here, so it is not created.
    from docusense.evaluation.qasper_loader import QASPERLoader

    harness = QASPERHarness.__new__(QASPERHarness)
    harness.loader = QASPERLoader()
    harness.dataset_path = TEST_SET
    chosen = harness.select_papers(papers, seed=SEED)
    sliced = {p.paper_id: test[p.paper_id] for p in chosen}

    sys.path.insert(0, str(ROOT / "scripts"))
    from validate_claim_verifier import build_examples

    dev = json.loads(DEV_SET.read_text(encoding="utf-8"))
    examples = build_examples(dev, SEED)
    rng = random.Random(SEED)
    supported = [e for e in examples if e[2] == 1]
    unsupported = [e for e in examples if e[2] == 0]
    picked = rng.sample(supported, claims_per_label) + rng.sample(unsupported, claims_per_label)
    claims = [
        {"claim": c, "sources": s, "supported": bool(y), "kind": k}
        for c, s, y, k in picked
    ]

    FIXTURES.mkdir(parents=True, exist_ok=True)
    RETRIEVAL_FIXTURE.write_text(json.dumps(sliced), encoding="utf-8")
    CLAIMS_FIXTURE.write_text(json.dumps(claims, indent=1), encoding="utf-8")
    print(f"{len(sliced)} papers -> {RETRIEVAL_FIXTURE.relative_to(ROOT)}")
    print(f"{len(claims)} claims -> {CLAIMS_FIXTURE.relative_to(ROOT)}")


def measure_retrieval(work: Path) -> dict:
    from docusense.config.settings import settings
    from docusense.evaluation.qasper_harness import DEFAULT_ARMS, QASPERHarness

    harness = QASPERHarness(dataset_path=RETRIEVAL_FIXTURE, work_dir=work / "papers")
    papers = len(json.loads(RETRIEVAL_FIXTURE.read_text(encoding="utf-8")))
    samples, stats = harness.prepare(num_papers=papers, seed=SEED)
    if stats.papers_ingested < papers:
        raise SystemExit(
            f"ERROR: {stats.papers_ingested} of {papers} fixture papers ingested; "
            f"the retrieval numbers would describe a different corpus"
        )

    metrics = {"retrieval.questions": len(samples)}
    for arm in DEFAULT_ARMS:
        if arm.name not in ("hybrid", "hybrid_rerank"):
            continue
        result = harness.run_arm(arm, samples, top_k=10).to_dict(include_per_query=False)
        m = result["retrieval_metrics"]
        metrics[f"retrieval.{arm.name}.mrr"] = round(m["MRR"], 4)
        metrics[f"retrieval.{arm.name}.recall@10"] = round(m["Recall@10"], 4)
    metrics["retrieval.embedding_model"] = settings.embedding_model
    return metrics


def measure_claims() -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    from validate_claim_verifier import roc_auc

    from docusense.config.settings import settings
    from docusense.generation.claim_verifier import ClaimVerifier, NLIModel

    claims = json.loads(CLAIMS_FIXTURE.read_text(encoding="utf-8"))
    threshold = settings.claim_support_threshold
    verifier = ClaimVerifier(model=NLIModel(), threshold=threshold)

    scores = [
        verifier.verify(c["claim"], c["sources"], claims=[c["claim"]]).claims[0].entailment
        for c in claims
    ]
    labels = [1 if c["supported"] else 0 for c in claims]
    flagged = [s < threshold for s in scores]
    supported_flagged = sum(f for f, y in zip(flagged, labels) if y) / max(1, sum(labels))
    unsupported_caught = sum(f for f, y in zip(flagged, labels) if not y) / max(
        1, len(labels) - sum(labels)
    )
    return {
        "claims.count": len(claims),
        "claims.roc_auc": round(roc_auc(scores, labels), 4),
        "claims.unsupported_caught": round(unsupported_caught, 4),
        "claims.supported_flagged": round(supported_flagged, 4),
        "claims.threshold": threshold,
    }


def compare(current: dict, baseline: dict) -> list[str]:
    """Every gated metric that moved the wrong way by more than its tolerance."""
    failures = []
    tolerance = baseline.get("tolerance", DEFAULT_TOLERANCE)

    # A baseline measured with another model or threshold is not a baseline
    # for this one; comparing them would pass or fail by accident.
    for key in ("retrieval.embedding_model", "claims.threshold",
                "retrieval.questions", "claims.count"):
        if key in baseline["metrics"] and current.get(key) != baseline["metrics"][key]:
            failures.append(
                f"{key} is {current.get(key)!r}, the baseline was measured with "
                f"{baseline['metrics'][key]!r}. Run --update-baseline if that is intended."
            )
    if failures:
        return failures

    for key, expected in baseline["metrics"].items():
        if not isinstance(expected, float) or key not in current:
            continue
        got = current[key]
        worse = got - expected if key in LOWER_IS_BETTER else expected - got
        if worse > tolerance:
            failures.append(
                f"{key}: {got:.4f} against a baseline of {expected:.4f} "
                f"({'+' if key in LOWER_IS_BETTER else '-'}{worse:.4f}, tolerance {tolerance})"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--build-fixtures", action="store_true")
    parser.add_argument("--papers", type=int, default=10)
    parser.add_argument("--claims-per-label", type=int, default=60)
    parser.add_argument("--only", choices=["retrieval", "claims"], default=None)
    args = parser.parse_args()

    if args.build_fixtures:
        build_fixtures(args.papers, args.claims_per_label)
        return 0

    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    started = time.time()
    current: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        if args.only != "claims":
            current.update(measure_retrieval(work))
        if args.only != "retrieval":
            current.update(measure_claims())
    seconds = time.time() - started

    print(json.dumps(current, indent=2))
    print(f"({seconds:.0f}s)")

    if args.update_baseline:
        old = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
        metrics = {**old.get("metrics", {}), **current}
        BASELINE.write_text(json.dumps({
            "tolerance": old.get("tolerance", DEFAULT_TOLERANCE),
            "measured": time.strftime("%Y-%m-%d"),
            "metrics": metrics,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Baseline written: {BASELINE.relative_to(ROOT)}")
        return 0

    if not BASELINE.exists():
        print("ERROR: no baseline; run with --update-baseline first", file=sys.stderr)
        return 1
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    if args.only:
        baseline["metrics"] = {
            k: v for k, v in baseline["metrics"].items() if k.startswith(args.only)
        }
    failures = compare(current, baseline)
    if failures:
        print("\nQUALITY GATE FAILED", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nQuality gate passed.")
    return 0


if __name__ == "__main__":
    # The gate builds its own corpus; it must never write into a developer's
    # store or the benchmark's. Set before anything imports the settings.
    os.environ["QDRANT_MODE"] = "memory"
    os.environ.setdefault("SQLITE_DB_PATH", str(Path(tempfile.gettempdir()) / f"docusense-gate-{os.getpid()}.db"))
    raise SystemExit(main())
