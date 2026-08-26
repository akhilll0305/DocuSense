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
        """Create the users table and its index."""
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
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

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> User:
        return User(
            user_id=row["user_id"],
            email=row["email"],
            name=row["name"],
            password_hash=row["password_hash"],
            created_at=row["created_at"],
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
