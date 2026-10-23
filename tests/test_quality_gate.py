"""The quality gate's verdict, without running any model."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("quality_gate", ROOT / "scripts" / "quality_gate.py")
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


BASELINE = {
    "tolerance": 0.02,
    "metrics": {
        "retrieval.questions": 31,
        "retrieval.embedding_model": "all-MiniLM-L6-v2",
        "retrieval.hybrid_rerank.mrr": 0.4621,
        "claims.count": 120,
        "claims.threshold": 0.5,
        "claims.roc_auc": 0.6981,
        "claims.supported_flagged": 0.3833,
    },
}


def run(**changes):
    return gate.compare({**BASELINE["metrics"], **changes}, BASELINE)


def test_identical_numbers_pass():
    assert run() == []


def test_drift_inside_the_tolerance_passes():
    assert run(**{"retrieval.hybrid_rerank.mrr": 0.4521, "claims.supported_flagged": 0.40}) == []


def test_improvements_pass():
    assert run(**{"claims.roc_auc": 0.90, "claims.supported_flagged": 0.10}) == []


def test_a_retrieval_regression_fails():
    failures = run(**{"retrieval.hybrid_rerank.mrr": 0.40})
    assert len(failures) == 1 and "hybrid_rerank.mrr" in failures[0]


def test_more_supported_claims_flagged_fails():
    # Lower is better for this one: a checker that starts warning on good
    # answers is a regression even though the number went up.
    failures = run(**{"claims.supported_flagged": 0.50})
    assert len(failures) == 1 and "supported_flagged" in failures[0]


@pytest.mark.parametrize("key,value", [
    ("retrieval.embedding_model", "BAAI/bge-small-en-v1.5"),
    ("claims.threshold", 0.8),
    ("claims.count", 60),
])
def test_a_different_setup_is_not_compared(key, value):
    # Numbers from another model or threshold are not comparable; the gate
    # says so instead of passing or failing by accident.
    failures = run(**{key: value})
    assert len(failures) == 1 and "--update-baseline" in failures[0]


def test_the_committed_baseline_covers_every_gated_metric():
    baseline = json.loads(gate.BASELINE.read_text(encoding="utf-8"))
    for key in ("retrieval.hybrid.mrr", "retrieval.hybrid_rerank.mrr",
                "claims.roc_auc", "claims.supported_flagged"):
        assert key in baseline["metrics"]
