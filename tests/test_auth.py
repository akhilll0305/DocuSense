"""
Authentication and tenant isolation tests.

Two layers:
1. Credential handling — hashing, tokens, register/login endpoints
2. Isolation — one user must never see, query, or delete another's data

The isolation tests run real components against an in-memory Qdrant, because
scoping bugs live in the wiring between SQLite, BM25, and the vector store,
which mocks would hide.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from docusense.auth import (
    AuthError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from docusense.config.settings import settings


# ==============================================================================
# PASSWORD HASHING
# ==============================================================================

def test_password_round_trip():
    hashed = hash_password("correct horse battery staple")
    assert hashed != "correct horse battery staple"
    assert verify_password("correct horse battery staple", hashed)


def test_wrong_password_rejected():
    hashed = hash_password("correct horse battery staple")
    assert not verify_password("Correct horse battery staple", hashed)
    assert not verify_password("", hashed)


def test_hashes_are_salted():
    """Identical passwords must not produce identical hashes."""
    assert hash_password("same password") != hash_password("same password")


def test_empty_password_rejected():
    with pytest.raises(AuthError):
        hash_password("")


def test_overlong_password_rejected():
    """bcrypt silently truncates past 72 bytes, so reject rather than mislead."""
    with pytest.raises(AuthError):
        hash_password("x" * 73)


def test_malformed_hash_does_not_raise():
    assert verify_password("anything", "not-a-bcrypt-hash") is False


# ==============================================================================
# TOKENS
# ==============================================================================

def test_token_round_trip():
    token = create_access_token("usr_123", "a@example.com")
    claims = decode_access_token(token)
    assert claims["sub"] == "usr_123"
    assert claims["email"] == "a@example.com"


def test_expired_token_rejected():
    token = create_access_token("usr_123", "a@example.com", expires_delta=timedelta(seconds=-1))
    with pytest.raises(AuthError, match="expired"):
        decode_access_token(token)


def test_tampered_token_rejected():
    token = create_access_token("usr_123", "a@example.com")
    with pytest.raises(AuthError):
        decode_access_token(token[:-4] + "AAAA")


def test_token_signed_with_other_key_rejected():
    """A token from a different deployment must not be accepted."""
    import jwt as pyjwt

    forged = pyjwt.encode({"sub": "usr_evil"}, "some-other-secret", algorithm="HS256")
    with pytest.raises(AuthError):
        decode_access_token(forged)


# ==============================================================================
# ISOLATION
# ==============================================================================

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """App backed by a temp database and in-memory Qdrant."""
    tmp = tmp_path_factory.mktemp("docusense_auth")

    original = (
        settings.sqlite_db_path,
        settings.qdrant_mode,
        settings.qdrant_url,
        settings.qdrant_api_key,
        settings.qdrant_collection_name,
    )
    settings.sqlite_db_path = tmp / "auth.db"
    settings.qdrant_mode = "memory"
    settings.qdrant_url = None
    settings.qdrant_api_key = None
    settings.qdrant_collection_name = "auth_chunks"

    from docusense.api.app import app

    with TestClient(app) as c:
        yield c

    (
        settings.sqlite_db_path,
        settings.qdrant_mode,
        settings.qdrant_url,
        settings.qdrant_api_key,
        settings.qdrant_collection_name,
    ) = original


def _register(client, email: str) -> dict:
    r = client.post(
        "/api/auth/register",
        json={"email": email, "password": "a-sufficiently-long-password", "name": email.split("@")[0]},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    return {"Authorization": f"Bearer {body['access_token']}"}


def _upload(client, headers, name: str, text: str) -> str:
    r = client.post(
        "/api/ingest",
        files={"file": (name, text.encode("utf-8"), "text/markdown")},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    assert r.json()["success"], r.text
    return r.json()["document_id"]


@pytest.fixture(scope="module")
def two_tenants(client):
    """Two users, each owning one document with distinctive content."""
    alice = _register(client, "alice@example.com")
    bob = _register(client, "bob@example.com")

    alice_doc = _upload(
        client, alice, "alice.md",
        "## Alice Notes\nThe zephyr protocol calibrates quantum flywheels at 4200 rpm.\n",
    )
    bob_doc = _upload(
        client, bob, "bob.md",
        "## Bob Notes\nThe marmalade index tracks citrus futures across 12 markets.\n",
    )
    return alice, bob, alice_doc, bob_doc


@pytest.mark.integration
def test_endpoints_require_authentication(client):
    """Every data endpoint must reject an unauthenticated caller."""
    assert client.get("/api/documents").status_code == 401
    assert client.get("/api/chats").status_code == 401
    assert client.post("/api/ask", json={"query": "hello"}).status_code == 401
    assert client.post("/api/chat/start", json={"title": "x"}).status_code == 401


@pytest.mark.integration
def test_health_stays_public(client):
    """Health checks must not require a token, or probes cannot reach them."""
    assert client.get("/api/health").status_code == 200


@pytest.mark.integration
def test_users_only_see_their_own_documents(client, two_tenants):
    alice, bob, alice_doc, bob_doc = two_tenants

    alice_ids = {d["document_id"] for d in client.get("/api/documents", headers=alice).json()["documents"]}
    bob_ids = {d["document_id"] for d in client.get("/api/documents", headers=bob).json()["documents"]}

    assert alice_doc in alice_ids and bob_doc not in alice_ids
    assert bob_doc in bob_ids and alice_doc not in bob_ids


@pytest.mark.integration
def test_retrieval_does_not_cross_tenants(client, two_tenants):
    """
    Alice querying Bob's distinctive text must retrieve nothing.

    This is the check that matters: the vector store is shared, so a missing
    user_id filter would surface Bob's chunks to Alice.
    """
    alice, bob, _, _ = two_tenants

    r = client.post(
        "/api/ask",
        json={"query": "marmalade index citrus futures", "top_k": 5},
        headers=alice,
    )
    assert r.status_code == 200
    for source in r.json()["sources"]:
        assert "marmalade" not in source.get("text_preview", "").lower()


@pytest.mark.integration
def test_user_cannot_delete_another_users_document(client, two_tenants):
    alice, bob, alice_doc, bob_doc = two_tenants

    # Alice tries to delete Bob's document; 404 hides its existence.
    assert client.delete(f"/api/documents/{bob_doc}", headers=alice).status_code == 404

    # Bob's document survives.
    bob_ids = {d["document_id"] for d in client.get("/api/documents", headers=bob).json()["documents"]}
    assert bob_doc in bob_ids


@pytest.mark.integration
def test_user_cannot_read_another_users_conversation(client, two_tenants):
    alice, bob, _, _ = two_tenants

    conv = client.post("/api/chat/start", json={"title": "Alice private"}, headers=alice).json()
    conv_id = conv["conversation_id"]

    assert client.get(f"/api/chat/{conv_id}", headers=bob).status_code == 404
    assert client.post(f"/api/chat/{conv_id}", json={"query": "hi"}, headers=bob).status_code == 404
    assert client.get(f"/api/chat/{conv_id}", headers=alice).status_code == 200


@pytest.mark.integration
def test_conversation_lists_are_scoped(client, two_tenants):
    alice, bob, _, _ = two_tenants
    client.post("/api/chat/start", json={"title": "Alice only"}, headers=alice)

    bob_titles = {c["title"] for c in client.get("/api/chats", headers=bob).json()}
    assert "Alice only" not in bob_titles
