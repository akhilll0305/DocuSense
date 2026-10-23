"""Readers' verdicts on the claim checker: storage, labels and the API."""

import pytest

from docusense.storage.feedback_store import FeedbackStore


@pytest.fixture
def store(tmp_path):
    s = FeedbackStore(db_path=tmp_path / "fb.db")
    yield s
    s.close()


def test_an_empty_store_has_no_precision(store):
    summary = store.summary()
    assert summary["flags_reviewed"] == 0
    assert summary["flag_precision"] is None


def test_flag_precision_is_the_share_of_flags_readers_confirmed(store):
    store.add("u1", "BERT has 340M parameters.", flagged=True, verdict="agree")
    store.add("u1", "It was trained on Wikipedia.", flagged=True, verdict="agree")
    store.add("u2", "The model uses 12 layers.", flagged=True, verdict="disagree")
    summary = store.summary()
    assert summary["flags_reviewed"] == 3
    assert summary["flags_confirmed"] == 2
    assert summary["flag_precision"] == pytest.approx(2 / 3, abs=1e-4)


def test_voting_again_replaces_the_earlier_vote(store):
    store.add("u1", "BERT has 340M parameters.", flagged=True, verdict="agree")
    store.add("u1", "  bert has 340M   parameters. ", flagged=True, verdict="disagree")
    summary = store.summary()
    assert summary["flags_reviewed"] == 1
    assert summary["flags_disputed"] == 1


def test_two_readers_on_one_statement_both_count(store):
    store.add("u1", "BERT has 340M parameters.", flagged=True, verdict="agree")
    store.add("u2", "BERT has 340M parameters.", flagged=True, verdict="agree")
    assert store.summary()["flags_reviewed"] == 2


def test_an_unknown_verdict_is_refused(store):
    with pytest.raises(ValueError):
        store.add("u1", "x", flagged=True, verdict="maybe")


@pytest.mark.parametrize("flagged,verdict,supported", [
    (True, "agree", False),      # checker said unsupported, reader agrees
    (True, "disagree", True),    # checker said unsupported, reader says it is there
    (False, "agree", True),
    (False, "disagree", False),  # checker passed it, reader says it is not there
])
def test_export_labels_follow_the_reader(store, flagged, verdict, supported):
    store.add("u1", "A statement.", flagged=flagged, verdict=verdict, evidence="A passage.")
    [example] = store.export()
    assert example["supported"] is supported
    assert example["sources"] == ["A passage."]
    assert example["kind"] == "reader_feedback"


def test_the_summary_carries_no_text(store):
    store.add("u1", "A very specific statement.", flagged=True, verdict="agree")
    assert "specific" not in repr(store.summary())


# ── API ──

@pytest.fixture
def client(tmp_path):
    from unittest.mock import MagicMock

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from docusense.api.deps import get_current_user, get_rag
    from docusense.api.routes import router
    from docusense.auth import User

    real = FeedbackStore(db_path=tmp_path / "api.db")
    rag = MagicMock()
    rag.record_claim_feedback.side_effect = lambda user_id, **kw: real.add(user_id=user_id, **kw)
    rag.claim_feedback_summary.side_effect = real.summary
    rag.owns_conversation.side_effect = lambda conv, user: conv == "conv_mine"

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_rag] = lambda: rag
    app.dependency_overrides[get_current_user] = lambda: User(
        user_id="usr_test", email="t@example.com", name="Test",
        password_hash="x", created_at="now",
    )
    yield TestClient(app)
    real.close()


def test_a_verdict_is_recorded_and_counted(client):
    r = client.post("/api/feedback/claim", json={
        "claim": "BERT has 340M parameters.", "verdict": "agree",
        "flagged": True, "entailment": 0.12, "conversation_id": "conv_mine",
    })
    assert r.status_code == 200 and r.json()["feedback_id"].startswith("fb_")
    summary = client.get("/api/feedback/claims").json()
    assert summary["flags_confirmed"] == 1 and summary["flag_precision"] == 1.0


def test_feedback_on_someone_elses_conversation_is_refused(client):
    r = client.post("/api/feedback/claim", json={
        "claim": "x y z w", "verdict": "agree", "conversation_id": "conv_theirs",
    })
    assert r.status_code == 404


@pytest.mark.parametrize("body", [
    {"claim": "", "verdict": "agree"},
    {"claim": "a claim", "verdict": "maybe"},
    {"claim": "a claim", "verdict": "agree", "entailment": 1.5},
])
def test_malformed_feedback_is_rejected(client, body):
    assert client.post("/api/feedback/claim", json=body).status_code == 422
