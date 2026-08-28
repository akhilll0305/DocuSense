"""
Seed the demo account for a public instance.

A deployed DocuSense with an empty shelf shows nothing: the visitor has to
register and find a paper before anything happens. This creates one account
with a couple of papers already ingested, so the first click lands on a working
system.

It is idempotent, and it is meant to run on every container start. Free hosting
tiers give ephemeral storage — a restart wipes the database — so seeding has to
be something the instance does to itself rather than something someone did once.

Nothing here runs unless asked: SEED_DEMO must be true, or --force passed. A
local install should not grow an account with a published password in it.

Usage:
    SEED_DEMO=true python scripts/seed_demo.py
    python scripts/seed_demo.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from loguru import logger  # noqa: E402

from docusense.auth import DuplicateEmailError, UserStore, hash_password  # noqa: E402
from docusense.config.settings import settings  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parent.parent / "data" / "demo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed the public demo account.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Seed even when SEED_DEMO is not set",
    )
    parser.add_argument(
        "--papers",
        default=str(DEMO_DIR),
        help=f"Directory of documents to ingest (default: {DEMO_DIR})",
    )
    return parser.parse_args()


def ensure_account(store: UserStore) -> str:
    """
    Create the demo account if it is missing, and return its user id.

    The password is deliberately a published one. This account exists to be
    shared; it is not a back door, because it owns nothing but the demo papers
    and per-user isolation keeps it out of everyone else's documents.
    """
    existing = store.get_by_email(settings.demo_email)
    if existing is not None:
        logger.info(f"Demo account already present: {existing.user_id}")
        return existing.user_id

    try:
        user = store.create_user(
            email=settings.demo_email,
            name="Demo reader",
            password_hash=hash_password(settings.demo_password),
        )
    except DuplicateEmailError:
        # Two containers starting at once; the other won.
        user = store.get_by_email(settings.demo_email)
        if user is None:
            raise
    logger.info(f"Created demo account {user.user_id}")
    return user.user_id


def main() -> int:
    args = parse_args()

    if not (settings.seed_demo or args.force):
        print("SEED_DEMO is not set and --force was not passed; nothing to do.")
        return 0

    papers_dir = Path(args.papers)
    if not papers_dir.is_dir():
        print(f"No such directory: {papers_dir}", file=sys.stderr)
        return 1

    store = UserStore()
    try:
        user_id = ensure_account(store)
    finally:
        store.close()

    from docusense.rag_pipeline import DocuSenseRAG

    rag = DocuSenseRAG(enable_images=False)
    try:
        already = {d["filename"] for d in rag.list_documents(user_id=user_id)}
        candidates = sorted(
            p for p in papers_dir.iterdir()
            if p.suffix.lower() in {".md", ".txt", ".pdf", ".docx"}
        )

        ingested = 0
        for path in candidates:
            if path.name in already:
                logger.info(f"Already on the demo shelf: {path.name}")
                continue
            result = rag.ingest(path, user_id=user_id, original_filename=path.name)
            if result.success:
                ingested += 1
                logger.info(f"Seeded {path.name}: {result.num_chunks} chunks")
            else:
                # One bad document must not stop the instance from starting.
                logger.warning(f"Could not seed {path.name}: {result.error}")

        total = len(rag.list_documents(user_id=user_id))
        print(
            f"Demo shelf ready: {total} document(s) for {settings.demo_email} "
            f"({ingested} added this run)."
        )
        return 0
    finally:
        rag.close()


if __name__ == "__main__":
    raise SystemExit(main())
