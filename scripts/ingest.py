"""
Ingest documents into DocuSense.

Runs the full pipeline for each file: convert -> chunk -> embed -> store in
SQLite and Qdrant.

Usage:
    python scripts/ingest.py path/to/paper.pdf
    python scripts/ingest.py data/demo/                    # whole directory
    python scripts/ingest.py data/papers/ --images         # describe figures too
    python scripts/ingest.py --reset data/papers/          # wipe the store first
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Result strings contain emoji; the default Windows console codepage (cp1252)
# raises UnicodeEncodeError on them.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".pptx", ".xlsx"}


def collect_files(targets: list[str]) -> list[Path]:
    """Expand paths and directories into a sorted list of ingestible files."""
    files: list[Path] = []
    for raw in targets:
        p = Path(raw)
        if p.is_dir():
            files.extend(f for f in sorted(p.rglob("*")) if f.suffix.lower() in SUPPORTED)
        elif p.is_file():
            if p.suffix.lower() not in SUPPORTED:
                print(f"Skipping unsupported file type: {p}")
                continue
            files.append(p)
        else:
            print(f"Path not found: {p}")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into DocuSense")
    parser.add_argument("paths", nargs="+", help="Files or directories to ingest")
    parser.add_argument(
        "--images", action="store_true",
        help="Describe figures with a vision model (slow, needs GEMINI_API_KEY)",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete the Qdrant collection before ingesting",
    )
    args = parser.parse_args()

    files = collect_files(args.paths)
    if not files:
        print("No ingestible files found.")
        return 1

    from docusense.rag_pipeline import DocuSenseRAG

    rag = DocuSenseRAG(enable_images=args.images)

    if args.reset:
        try:
            rag.qdrant_store.delete_collection()
            print("Deleted existing Qdrant collection.")
        except Exception as e:
            print(f"Could not delete collection (may not exist yet): {e}")
        rag.qdrant_store.create_collection()

    print(f"\nIngesting {len(files)} file(s)...\n")

    succeeded, failed, chunks = 0, 0, 0
    try:
        for f in files:
            result = rag.ingest(f)
            print(result)
            if result.success:
                succeeded += 1
                chunks += result.num_chunks
            else:
                failed += 1
    finally:
        rag.close()

    print(f"\n{succeeded} succeeded, {failed} failed, {chunks} chunks total")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
