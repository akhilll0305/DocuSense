"""
Session revocation and password change.

A signed JWT is valid until it expires, so "log out" without server-side state
is only a client-side gesture — the token keeps working anywhere else it was
copied. These tests exercise the two mechanisms that make it real: a blocklist
of individual token ids, and a per-user token version that invalidates every
outstanding token in one write.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from docusense.auth import decode_access_token
from docusense.config.settings import settings


PASSWORD = "a-sufficiently-long-password"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """App backed by a temp database and in-memory Qdrant."""
    tmp = tmp_path_factory.mktemp("docusense_revocation")

    original = (
        settings.sqlite_db_path,
        settings.qdrant_mode,
        settings.qdrant_url,
        settings.qdrant_api_key,
        settings.qdrant_collection_name,
    )
    settings.sqlite_db_path = tmp / "revocation.db"
    settings.qdrant_mode = "memory"
    settings.qdrant_url = None
    settings.qdrant_api_key = None
    settings.qdrant_collection_name = "revocation_chunks"

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


def _register(client, email: str) -> str:
    r = client.post(
        "/api/auth/register",
        json={"email": email, "password": PASSWORD, "name": email.split("@")[0]},
    )
    assert r.status_code == 201, r.text
    return r.json()["access_token"]


def _login(client, email: str, password: str = PASSWORD) -> str:
    r = client.post("/api/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ==============================================================================
# Token claims
# ==============================================================================

@pytest.mark.integration
def test_tokens_carry_an_id_and_a_version(client):
    """Revocation needs both: an id to blocklist, a version to compare."""
    token = _register(client, "claims@example.com")
    claims = decode_access_token(token)
    assert claims["jti"]
    assert claims["tv"] == 1


@pytest.mark.integration
def test_two_logins_get_different_token_ids(client):
    """Otherwise signing out one device would sign out all of them."""
    _register(client, "two-ids@example.com")
    a = decode_access_token(_login(client, "two-ids@example.com"))
    b = decode_access_token(_login(client, "two-ids@example.com"))
    assert a["jti"] != b["jti"]


# ==============================================================================
# Signing out one session
# ==============================================================================

@pytest.mark.integration
def test_logout_makes_the_token_stop_working(client):
    """
    The defect this exists for: logout cleared the client's copy and nothing
    else, so a token captured anywhere kept working until it expired.
    """
    token = _register(client, "logout@example.com")
    assert client.get("/api/auth/me", headers=_auth(token)).status_code == 200

    assert client.post("/api/auth/logout", headers=_auth(token)).status_code == 200

    r = client.get("/api/auth/me", headers=_auth(token))
    assert r.status_code == 401
    assert "signed out" in r.json()["detail"].lower()


@pytest.mark.integration
def test_logout_leaves_other_sessions_alone(client):
    """Signing out of a phone must not sign out the laptop."""
    _register(client, "two-devices@example.com")
    phone = _login(client, "two-devices@example.com")
    laptop = _login(client, "two-devices@example.com")

    client.post("/api/auth/logout", headers=_auth(phone))

    assert client.get("/api/auth/me", headers=_auth(phone)).status_code == 401
    assert client.get("/api/auth/me", headers=_auth(laptop)).status_code == 200


@pytest.mark.integration
def test_a_revoked_token_cannot_reach_data_endpoints(client):
    """The check belongs in the shared dependency, not in one route."""
    token = _register(client, "revoked-data@example.com")
    client.post("/api/auth/logout", headers=_auth(token))

    assert client.get("/api/documents", headers=_auth(token)).status_code == 401
    assert client.get("/api/chats", headers=_auth(token)).status_code == 401
    assert client.get("/api/stats", headers=_auth(token)).status_code == 401


# ==============================================================================
# Signing out everywhere
# ==============================================================================

@pytest.mark.integration
def test_logout_all_invalidates_every_session(client):
    _register(client, "everywhere@example.com")
    phone = _login(client, "everywhere@example.com")
    laptop = _login(client, "everywhere@example.com")

    assert client.post("/api/auth/logout-all", headers=_auth(phone)).status_code == 200

    assert client.get("/api/auth/me", headers=_auth(phone)).status_code == 401
    assert client.get("/api/auth/me", headers=_auth(laptop)).status_code == 401


@pytest.mark.integration
def test_logging_in_again_after_logout_all_works(client):
    """Signing out everywhere must not lock the account."""
    _register(client, "relogin@example.com")
    old = _login(client, "relogin@example.com")
    client.post("/api/auth/logout-all", headers=_auth(old))

    fresh = _login(client, "relogin@example.com")
    assert client.get("/api/auth/me", headers=_auth(fresh)).status_code == 200


@pytest.mark.integration
def test_logout_all_does_not_touch_other_accounts(client):
    _register(client, "mine@example.com")
    _register(client, "theirs@example.com")
    mine = _login(client, "mine@example.com")
    theirs = _login(client, "theirs@example.com")

    client.post("/api/auth/logout-all", headers=_auth(mine))

    assert client.get("/api/auth/me", headers=_auth(theirs)).status_code == 200


# ==============================================================================
# Password change
# ==============================================================================

@pytest.mark.integration
def test_password_change_requires_the_current_password(client):
    token = _register(client, "wrongcurrent@example.com")
    r = client.post(
        "/api/auth/password",
        json={"current_password": "not-the-password", "new_password": "a-brand-new-password"},
        headers=_auth(token),
    )
    assert r.status_code == 401
    # The old password must still work.
    assert client.get("/api/auth/me", headers=_auth(token)).status_code == 200


@pytest.mark.integration
def test_password_change_switches_the_credentials(client):
    _register(client, "change@example.com")
    token = _login(client, "change@example.com")

    r = client.post(
        "/api/auth/password",
        json={"current_password": PASSWORD, "new_password": "an-entirely-new-password"},
        headers=_auth(token),
    )
    assert r.status_code == 200, r.text

    assert client.post(
        "/api/auth/login",
        json={"email": "change@example.com", "password": PASSWORD},
    ).status_code == 401
    assert _login(client, "change@example.com", "an-entirely-new-password")


@pytest.mark.integration
def test_password_change_signs_out_the_other_sessions(client):
    """
    A password changed because it may have leaked is worth nothing while the
    tokens minted with the old one keep working.
    """
    _register(client, "leak@example.com")
    attacker = _login(client, "leak@example.com")
    owner = _login(client, "leak@example.com")

    r = client.post(
        "/api/auth/password",
        json={"current_password": PASSWORD, "new_password": "a-replacement-password"},
        headers=_auth(owner),
    )
    assert r.status_code == 200

    assert client.get("/api/auth/me", headers=_auth(attacker)).status_code == 401
    # The old token of the caller who made the change is dead too.
    assert client.get("/api/auth/me", headers=_auth(owner)).status_code == 401
    # The replacement it handed back works.
    assert client.get(
        "/api/auth/me", headers=_auth(r.json()["access_token"])
    ).status_code == 200


@pytest.mark.integration
def test_password_change_rejects_reuse_and_short_passwords(client):
    _register(client, "weak@example.com")
    token = _login(client, "weak@example.com")

    same = client.post(
        "/api/auth/password",
        json={"current_password": PASSWORD, "new_password": PASSWORD},
        headers=_auth(token),
    )
    assert same.status_code == 422

    short = client.post(
        "/api/auth/password",
        json={"current_password": PASSWORD, "new_password": "short"},
        headers=_auth(token),
    )
    assert short.status_code == 422


@pytest.mark.integration
def test_password_endpoints_reject_anonymous_callers(client):
    assert client.post("/api/auth/logout").status_code == 401
    assert client.post("/api/auth/logout-all").status_code == 401
    assert client.post(
        "/api/auth/password",
        json={"current_password": "x", "new_password": "a-long-enough-password"},
    ).status_code == 401


# ==============================================================================
# Schema migration
# ==============================================================================

@pytest.mark.integration
def test_migrates_a_database_created_before_revocation(tmp_path):
    """
    `CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists,
    so an existing database keeps a users table with no token_version column
    and every read of it raises. The store has to add the column.
    """
    import sqlite3

    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute(
        "INSERT INTO users (user_id, email, name, password_hash, created_at) "
        "VALUES ('usr_legacy', 'legacy@example.com', 'legacy', 'hash', '2026-01-01')"
    )
    conn.commit()
    conn.close()

    from docusense.auth import UserStore

    store = UserStore(db_path=db)
    try:
        user = store.get_by_id("usr_legacy")
        assert user is not None
        # Existing accounts start at version 1, so the tokens they already hold
        # keep working until something revokes them.
        assert user.token_version == 1
        assert store.bump_token_version("usr_legacy") == 2
    finally:
        store.close()


@pytest.mark.integration
def test_expired_revocations_are_purged(tmp_path):
    """The blocklist is bounded by token lifetime, not by how often people
    sign out."""
    from datetime import datetime, timezone

    from docusense.auth import UserStore

    store = UserStore(db_path=tmp_path / "purge.db")
    try:
        now = int(datetime.now(timezone.utc).timestamp())
        store.revoke_token("stale", "usr_x", now - 3600)
        assert store.is_token_revoked("stale")

        # Any later revocation trims what has since expired.
        store.revoke_token("fresh", "usr_x", now + 3600)
        assert not store.is_token_revoked("stale")
        assert store.is_token_revoked("fresh")
    finally:
        store.close()
