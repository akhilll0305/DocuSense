"""
Readers' verdicts on the claim checker's flags.

The claim checker was validated before it shipped, on labels built from
QASPER (scripts/validate_claim_verifier.py): about 64% of what it flags is
really unsupported. That is a measurement on annotators' answers about NLP
papers, not on this product's answers about whatever a reader uploads, and the
two can differ. The reader is the one person who can tell: they have the paper
open.

Each flagged statement in the UI asks whether the flag was right. The answer
is stored here with the checker's score and the passage it compared against,
which gives two things no offline benchmark can:

    a running estimate of flag precision on real use (`summary`), and
    labelled examples in the format the validation script reads (`export`),
    to re-tune the threshold or re-validate a new model on.

One vote per reader per statement: voting again replaces the earlier vote.
"""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from docusense.config.settings import settings

VERDICTS = ("agree", "disagree")


class FeedbackStore:
    """SQLite table of claim-flag verdicts, in the application database."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or settings.sqlite_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS claim_feedback (
                feedback_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                claim_hash TEXT NOT NULL,
                conversation_id TEXT,
                claim TEXT NOT NULL,
                evidence TEXT DEFAULT '',
                entailment REAL,
                flagged INTEGER NOT NULL,
                verdict TEXT NOT NULL CHECK(verdict IN ('agree', 'disagree')),
                created_at TEXT NOT NULL,
                UNIQUE (user_id, claim_hash)
            )
        """)
        self.conn.commit()

    @staticmethod
    def _hash(claim: str) -> str:
        return hashlib.sha256(" ".join(claim.split()).lower().encode("utf-8")).hexdigest()

    def add(
        self,
        user_id: str,
        claim: str,
        flagged: bool,
        verdict: str,
        evidence: str = "",
        entailment: Optional[float] = None,
        conversation_id: Optional[str] = None,
    ) -> str:
        """Record (or replace) one reader's verdict on one statement."""
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")
        feedback_id = f"fb_{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            """
            INSERT INTO claim_feedback
                (feedback_id, user_id, claim_hash, conversation_id, claim, evidence,
                 entailment, flagged, verdict, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (user_id, claim_hash) DO UPDATE SET
                verdict = excluded.verdict,
                evidence = excluded.evidence,
                entailment = excluded.entailment,
                flagged = excluded.flagged,
                created_at = excluded.created_at
            """,
            (feedback_id, user_id, self._hash(claim), conversation_id, claim,
             evidence or "", entailment, int(bool(flagged)), verdict,
             datetime.now().isoformat()),
        )
        self.conn.commit()
        return feedback_id

    def summary(self) -> Dict[str, Any]:
        """
        How often readers agreed with the checker, across every reader.

        Counts only: no statement text and no user leaves this method, so it
        is safe to show anyone.
        """
        rows = self.conn.execute(
            "SELECT flagged, verdict, COUNT(*) AS n FROM claim_feedback "
            "GROUP BY flagged, verdict"
        ).fetchall()
        counts = {(bool(r["flagged"]), r["verdict"]): r["n"] for r in rows}
        confirmed = counts.get((True, "agree"), 0)
        disputed = counts.get((True, "disagree"), 0)
        reviewed = confirmed + disputed
        return {
            "flags_reviewed": reviewed,
            "flags_confirmed": confirmed,
            "flags_disputed": disputed,
            # The offline estimate was 0.64; this is the same quantity on real use.
            "flag_precision": round(confirmed / reviewed, 4) if reviewed else None,
            "passes_reviewed": counts.get((False, "agree"), 0) + counts.get((False, "disagree"), 0),
            "passes_disputed": counts.get((False, "disagree"), 0),
        }

    def export(self) -> List[Dict[str, Any]]:
        """
        Every verdict as a labelled example, in tests/quality/claims.json's format.

        `supported` is what the reader says is true: a confirmed flag is an
        unsupported claim, a disputed flag a supported one. The source is the
        closest passage the checker found, not the whole paper, so a disputed
        flag whose support is elsewhere in the paper will still read as
        unsupported against it -- worth knowing before tuning on these.
        """
        rows = self.conn.execute(
            "SELECT claim, evidence, entailment, flagged, verdict, created_at "
            "FROM claim_feedback ORDER BY created_at"
        ).fetchall()
        out = []
        for r in rows:
            reader_says_supported = (r["verdict"] == "disagree") == bool(r["flagged"])
            out.append({
                "claim": r["claim"],
                "sources": [r["evidence"]] if r["evidence"] else [],
                "supported": reader_says_supported,
                "kind": "reader_feedback",
                "entailment": r["entailment"],
                "created_at": r["created_at"],
            })
        return out

    def close(self) -> None:
        self.conn.close()


__all__ = ["FeedbackStore", "VERDICTS"]
