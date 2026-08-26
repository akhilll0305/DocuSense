"""
Conversation Store - SQLite storage for conversations and query history.

Phase 5: Complete RAG Pipeline (Step 1)

PURPOSE:
--------
Persistent storage for multi-turn conversations and query tracking:
1. Conversations (title, creation time, metadata)
2. Messages (user/assistant turns with sources)
3. Query history (performance tracking and analytics)

Extends the existing SQLite storage pattern from chunk_store.py.

Author: DocuSense
Created: 2026-03-08
"""

import sqlite3
import json
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger

from docusense.config.settings import settings


@dataclass
class Conversation:
    """A conversation session."""
    conversation_id: str
    title: str = "New Conversation"
    created_at: str = ""
    updated_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


@dataclass
class Message:
    """A single message in a conversation."""
    message_id: str
    conversation_id: str
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class QueryLog:
    """A logged query for analytics."""
    query_id: str
    query: str
    processed_query: str = ""
    filters: Dict[str, Any] = field(default_factory=dict)
    num_results: int = 0
    response_time: float = 0.0
    confidence: float = 0.0
    model_used: str = ""
    conversation_id: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ConversationStore:
    """
    SQLite storage for conversations and query history.

    Tables:
    - conversations: Session tracking
    - messages: Ordered message history per conversation
    - query_history: Analytics and performance tracking

    Usage:
        store = ConversationStore()
        conv_id = store.create_conversation("BERT Analysis")
        store.add_message(conv_id, "user", "What is BERT?")
        store.add_message(conv_id, "assistant", "BERT is...", sources=[...])
        history = store.get_messages(conv_id)
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize conversation store.

        Args:
            db_path: Path to SQLite DB (default from settings)
        """
        self.db_path = db_path or settings.sqlite_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

        logger.info(f"💬 ConversationStore initialized at {self.db_path}")

    def _create_schema(self):
        """Create conversation tables."""
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New Conversation',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        """)

        # Migration: conversations created before multi-tenancy have no owner.
        existing = {r["name"] for r in cursor.execute("PRAGMA table_info(conversations)")}
        if "user_id" not in existing:
            cursor.execute("ALTER TABLE conversations ADD COLUMN user_id TEXT")
            logger.info("Migrated conversations table: added user_id column")

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)"
        )

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                sources TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                    ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                query_id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                processed_query TEXT DEFAULT '',
                filters TEXT DEFAULT '{}',
                num_results INTEGER DEFAULT 0,
                response_time REAL DEFAULT 0.0,
                confidence REAL DEFAULT 0.0,
                model_used TEXT DEFAULT '',
                conversation_id TEXT,
                timestamp TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages(conversation_id, timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_query_history_timestamp
            ON query_history(timestamp DESC)
        """)

        self.conn.commit()

    # ==================================================================
    # CONVERSATION MANAGEMENT
    # ==================================================================

    def create_conversation(
        self,
        title: str = "New Conversation",
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> str:
        """
        Create a new conversation.

        Args:
            title: Display title
            metadata: Arbitrary JSON metadata
            user_id: Owning user; conversations are listed per owner

        Returns:
            conversation_id
        """
        conv_id = f"conv_{uuid.uuid4().hex[:12]}"
        now = datetime.now().isoformat()

        self.conn.execute(
            "INSERT INTO conversations "
            "(conversation_id, title, created_at, updated_at, metadata, user_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (conv_id, title, now, now, json.dumps(metadata or {}), user_id)
        )
        self.conn.commit()

        logger.info(f"💬 Created conversation: {conv_id} ({title})")
        return conv_id

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        row = self.conn.execute(
            "SELECT * FROM conversations WHERE conversation_id = ?",
            (conversation_id,)
        ).fetchone()

        if not row:
            return None

        return Conversation(
            conversation_id=row["conversation_id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {}
        )

    def list_conversations(
        self, limit: int = 20, user_id: Optional[str] = None
    ) -> List[Conversation]:
        """List recent conversations, optionally scoped to one owner."""
        if user_id is None:
            rows = self.conn.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM conversations WHERE user_id = ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (user_id, limit)
            ).fetchall()

        return [
            Conversation(
                conversation_id=row["conversation_id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {}
            )
            for row in rows
        ]

    def get_conversation_owner(self, conversation_id: str) -> Optional[str]:
        """Return the user_id owning a conversation, or None if unowned/missing."""
        row = self.conn.execute(
            "SELECT user_id FROM conversations WHERE conversation_id = ?",
            (conversation_id,)
        ).fetchone()
        return row["user_id"] if row else None

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages."""
        cursor = self.conn.execute(
            "DELETE FROM conversations WHERE conversation_id = ?",
            (conversation_id,)
        )
        self.conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"🗑️ Deleted conversation: {conversation_id}")
        return deleted

    # ==================================================================
    # MESSAGE MANAGEMENT
    # ==================================================================

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a message to a conversation.

        Returns:
            message_id
        """
        msg_id = f"msg_{uuid.uuid4().hex[:12]}"
        now = datetime.now().isoformat()

        self.conn.execute(
            "INSERT INTO messages (message_id, conversation_id, role, content, "
            "timestamp, sources, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (msg_id, conversation_id, role, content, now,
             json.dumps(sources or []), json.dumps(metadata or {}))
        )

        # Update conversation timestamp
        self.conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
            (now, conversation_id)
        )
        self.conn.commit()

        return msg_id

    def get_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[Message]:
        """
        Get messages for a conversation, ordered by timestamp.

        Args:
            conversation_id: Conversation to query
            limit: Optional max messages (most recent first if limited)
        """
        if limit:
            rows = self.conn.execute(
                "SELECT * FROM ("
                "  SELECT * FROM messages WHERE conversation_id = ? "
                "  ORDER BY timestamp DESC LIMIT ?"
                ") ORDER BY timestamp ASC",
                (conversation_id, limit)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC",
                (conversation_id,)
            ).fetchall()

        return [
            Message(
                message_id=row["message_id"],
                conversation_id=row["conversation_id"],
                role=row["role"],
                content=row["content"],
                timestamp=row["timestamp"],
                sources=json.loads(row["sources"]) if row["sources"] else [],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {}
            )
            for row in rows
        ]

    # ==================================================================
    # QUERY HISTORY
    # ==================================================================

    def log_query(
        self,
        query: str,
        processed_query: str = "",
        filters: Optional[Dict[str, Any]] = None,
        num_results: int = 0,
        response_time: float = 0.0,
        confidence: float = 0.0,
        model_used: str = "",
        conversation_id: Optional[str] = None
    ) -> str:
        """
        Log a query for analytics.

        Returns:
            query_id
        """
        query_id = f"qry_{uuid.uuid4().hex[:12]}"
        now = datetime.now().isoformat()

        self.conn.execute(
            "INSERT INTO query_history (query_id, query, processed_query, filters, "
            "num_results, response_time, confidence, model_used, conversation_id, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (query_id, query, processed_query, json.dumps(filters or {}),
             num_results, response_time, confidence, model_used,
             conversation_id, now)
        )
        self.conn.commit()

        return query_id

    def get_query_history(self, limit: int = 50) -> List[QueryLog]:
        """Get recent query history."""
        rows = self.conn.execute(
            "SELECT * FROM query_history ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()

        return [
            QueryLog(
                query_id=row["query_id"],
                query=row["query"],
                processed_query=row["processed_query"],
                filters=json.loads(row["filters"]) if row["filters"] else {},
                num_results=row["num_results"],
                response_time=row["response_time"],
                confidence=row["confidence"],
                model_used=row["model_used"],
                conversation_id=row["conversation_id"],
                timestamp=row["timestamp"]
            )
            for row in rows
        ]

    def get_query_stats(self) -> Dict[str, Any]:
        """Get aggregate query statistics."""
        cursor = self.conn.cursor()
        total = cursor.execute("SELECT COUNT(*) FROM query_history").fetchone()[0]

        if total == 0:
            return {"total_queries": 0}

        avg_time = cursor.execute(
            "SELECT AVG(response_time) FROM query_history"
        ).fetchone()[0]
        avg_confidence = cursor.execute(
            "SELECT AVG(confidence) FROM query_history WHERE confidence > 0"
        ).fetchone()[0]
        avg_results = cursor.execute(
            "SELECT AVG(num_results) FROM query_history"
        ).fetchone()[0]

        return {
            "total_queries": total,
            "avg_response_time": round(avg_time or 0, 3),
            "avg_confidence": round(avg_confidence or 0, 3),
            "avg_results": round(avg_results or 0, 1)
        }

    # ==================================================================
    # CLEANUP
    # ==================================================================

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
