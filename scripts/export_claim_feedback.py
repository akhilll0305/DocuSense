"""
Readers' verdicts on the claim checker, as labelled examples.

    python scripts/export_claim_feedback.py                 # summary only
    python scripts/export_claim_feedback.py --out feedback.json

The output is in the format of tests/quality/claims.json, so it can be scored
the same way: append it to the gate's fixture once there are enough verdicts,
or use it to check whether claim_support_threshold still sits in the right
place on real answers rather than on QASPER annotations.

Statements are what readers' answers contained, so the file is user data: it
is written where you point it and nowhere else, and is not for committing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description="Export claim-check feedback")
    parser.add_argument("--out", default=None, help="Write the labelled examples here")
    parser.add_argument("--db", default=None, help="SQLite path (default: settings)")
    args = parser.parse_args()

    from docusense.storage.feedback_store import FeedbackStore

    store = FeedbackStore(db_path=args.db)
    summary = store.summary()
    print(json.dumps(summary, indent=2))
    if summary["flag_precision"] is not None:
        print(f"\nReaders confirmed {summary['flag_precision']:.0%} of the flags they "
              f"reviewed (offline estimate on QASPER: 64%).")

    if args.out:
        examples = store.export()
        Path(args.out).write_text(json.dumps(examples, indent=1), encoding="utf-8")
        print(f"{len(examples)} labelled examples -> {args.out}")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
