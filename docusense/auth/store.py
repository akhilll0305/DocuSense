"""
User storage — SQLite-backed accounts for DocuSense.

Shares the application database with documents and chunks so that a document's
owner is enforceable with a foreign key.

Author: DocuSense
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from loguru import logger

from docusense.config.settings import settings


@dataclass
class User:
    """An application user. `password_hash` is never serialized to clients."""
    user_id: str
    email: str
    name: str
    password_hash: str
    created_at: str
    # Bumped to invalidate every token this user currently holds. Tokens carry
    # the version they were issued under and are rejected once it moves on.
    token_version: int = 1

    def public(self) -> dict:
        """Representation safe to return over the API."""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "name": self.name,
            "created_at": self.created_at,
        }


class DuplicateEmailError(Exception):
    """Raised when registering an email that already exists."""


class UserStore:
    """
    CRUD for user accounts.

    Emails are stored and compared lowercased so addresses differing only in
    case cannot create two accounts.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or settings.sqlite_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.create_schema()

    def create_schema(self) -> None:
        """Create the users and revoked-token tables, and their indexes."""
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                token_version INTEGER NOT NULL DEFAULT 1
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")

        # Tokens signed out individually. A JWT is valid until it expires, so
        # signing out has to be recorded somewhere the server can see; rows are
        # dropped once the token they name would have expired anyway, which
        # bounds the table by the token lifetime rather than by usage.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS revoked_tokens (
                jti TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                expires_at INTEGER NOT NULL,
                revoked_at TEXT NOT NULL
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_revoked_expires "
            "ON revoked_tokens(expires_at)"
        )
        self.conn.commit()

        self._migrate()

    def _migrate(self) -> None:
        """
        Bring an existing database up to the current schema.

        `CREATE TABLE IF NOT EXISTS` does nothing to a table that already
        exists, so a database created before token revocation has a users table
        with no token_version column and every read of it would raise.
        """
        columns = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(users)").fetchall()
        }
        if "token_version" not in columns:
            logger.info("Migrating users table: adding token_version")
            self.conn.execute(
                "ALTER TABLE users ADD COLUMN token_version INTEGER NOT NULL DEFAULT 1"
            )
            self.conn.commit()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def create_user(self, email: str, name: str, password_hash: str) -> User:
        """
        Insert a new user.

        Raises:
            DuplicateEmailError: if the email is already registered
        """
        email = email.strip().lower()
        user = User(
            user_id=f"usr_{uuid.uuid4().hex[:12]}",
            email=email,
            name=name.strip() or email.split("@")[0],
            password_hash=password_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        try:
            self.conn.execute(
                "INSERT INTO users (user_id, email, name, password_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user.user_id, user.email, user.name, user.password_hash, user.created_at),
            )
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            raise DuplicateEmailError(f"Email already registered: {email}") from e

        logger.info(f"Created user {user.user_id} ({user.email})")
        return user

    def set_password(self, user_id: str, password_hash: str) -> int:
        """
        Replace a user's password and invalidate their existing sessions.

        The two happen together on purpose. Changing a password because it may
        have leaked is pointless if the tokens minted with the old one keep
        working, so the token version moves in the same transaction.

        Args:
            user_id: Whose password to change
            password_hash: The new bcrypt hash

        Returns:
            The user's new token version

        Raises:
            KeyError: if no such user exists
        """
        cur = self.conn.execute(
            "UPDATE users SET password_hash = ?, token_version = token_version + 1 "
            "WHERE user_id = ?",
            (password_hash, user_id),
        )
        if cur.rowcount == 0:
            self.conn.rollback()
            raise KeyError(f"No such user: {user_id}")
        self.conn.commit()

        version = self.conn.execute(
            "SELECT token_version FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        logger.info(f"Password changed for {user_id}; token version now {version}")
        return version

    def bump_token_version(self, user_id: str) -> int:
        """
        Invalidate every token currently held by a user.

        One write covers every outstanding session, including ones this server
        has never seen, which a blocklist cannot do without knowing their ids.

        Returns:
            The new token version

        Raises:
            KeyError: if no such user exists
        """
        cur = self.conn.execute(
            "UPDATE users SET token_version = token_version + 1 WHERE user_id = ?",
            (user_id,),
        )
        if cur.rowcount == 0:
            self.conn.rollback()
            raise KeyError(f"No such user: {user_id}")
        self.conn.commit()

        version = self.conn.execute(
            "SELECT token_version FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        logger.info(f"Signed {user_id} out everywhere; token version now {version}")
        return version

    def revoke_token(self, jti: str, user_id: str, expires_at: int) -> None:
        """
        Record one token as signed out.

        Args:
            jti: The token's unique id
            user_id: Its subject, for auditing
            expires_at: The token's `exp`, as a unix timestamp. The row is
                dropped once this passes, because the token is worthless by
                then anyway.
        """
        self._purge_expired_revocations()
        self.conn.execute(
            "INSERT OR IGNORE INTO revoked_tokens (jti, user_id, expires_at, revoked_at) "
            "VALUES (?, ?, ?, ?)",
            (jti, user_id, int(expires_at), datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    def is_token_revoked(self, jti: str) -> bool:
        """Whether this specific token has been signed out."""
        if not jti:
            return False
        row = self.conn.execute(
            "SELECT 1 FROM revoked_tokens WHERE jti = ?", (jti,)
        ).fetchone()
        return row is not None

    def _purge_expired_revocations(self) -> int:
        """
        Drop revocation rows for tokens that have expired on their own.

        Runs on write rather than on a timer: the table only grows when someone
        signs out, so that is the only moment it can need trimming, and it
        keeps the store free of background threads.
        """
        now = int(datetime.now(timezone.utc).timestamp())
        cur = self.conn.execute(
            "DELETE FROM revoked_tokens WHERE expires_at < ?", (now,)
        )
        return cur.rowcount

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> User:
        keys = row.keys()
        return User(
            user_id=row["user_id"],
            email=row["email"],
            name=row["name"],
            password_hash=row["password_hash"],
            created_at=row["created_at"],
            token_version=row["token_version"] if "token_version" in keys else 1,
        )

    def get_by_email(self, email: str) -> Optional[User]:
        """Look up a user by email (case-insensitive)."""
        row = self.conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
        ).fetchone()
        return self._row_to_user(row) if row else None

    def get_by_id(self, user_id: str) -> Optional[User]:
        """Look up a user by their stable id."""
        row = self.conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        return self._row_to_user(row) if row else None

    def count(self) -> int:
        """Total registered users."""
        return self.conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def list_users(self) -> List[User]:
        """All users, newest first. Intended for administrative use."""
        rows = self.conn.execute(
            "SELECT * FROM users ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_user(r) for r in rows]

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
