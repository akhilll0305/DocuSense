"""
Reset a user's password from the machine running DocuSense.

This is the operator recovery path, and it is deliberately not a web endpoint.
Self-service "forgot password" needs a way to prove the person asking owns the
address — in practice a mailed one-time link — and DocuSense has no email
channel. An endpoint that reset a password on request alone would let anyone
take any account by knowing its email address, so it is not offered. Whoever
can run this script already has the database file and could edit the row by
hand; the script only makes it safe to do.

Resetting also signs the account out everywhere: a password changed because it
leaked is worth nothing while tokens minted with the old one keep working.

Usage:
    python scripts/reset_password.py user@example.com
    python scripts/reset_password.py user@example.com --generate
    python scripts/reset_password.py --list
"""

from __future__ import annotations

import argparse
import getpass
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from docusense.auth import AuthError, UserStore, hash_password  # noqa: E402
from docusense.config.settings import settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reset a DocuSense account password (local operator tool).",
    )
    parser.add_argument("email", nargs="?", help="Account to reset")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate a strong password and print it once, instead of prompting",
    )
    parser.add_argument(
        "--list", action="store_true", help="List accounts and exit"
    )
    return parser.parse_args()


def list_accounts(store: UserStore) -> int:
    users = store.list_users()
    if not users:
        print("No accounts registered.")
        return 0
    print(f"{len(users)} account(s):\n")
    for user in users:
        print(f"  {user.email:40s} {user.user_id}  created {user.created_at[:10]}")
    return 0


def read_new_password() -> str:
    """Prompt twice, without echoing, and check the two agree."""
    first = getpass.getpass("New password: ")
    second = getpass.getpass("Repeat it: ")
    if first != second:
        print("Passwords do not match.", file=sys.stderr)
        raise SystemExit(1)
    return first


def main() -> int:
    args = parse_args()
    store = UserStore()

    try:
        if args.list:
            return list_accounts(store)

        if not args.email:
            print("Give an email address, or --list to see the accounts.",
                  file=sys.stderr)
            return 2

        user = store.get_by_email(args.email)
        if user is None:
            # No enumeration concern here: whoever runs this already has the
            # database. Being vague would only waste the operator's time.
            print(f"No account for {args.email}.", file=sys.stderr)
            return 1

        if args.generate:
            password = secrets.token_urlsafe(18)
        else:
            password = read_new_password()

        if len(password) < settings.min_password_length:
            print(
                f"Password must be at least {settings.min_password_length} "
                f"characters.",
                file=sys.stderr,
            )
            return 1

        try:
            new_hash = hash_password(password)
        except AuthError as e:
            print(f"{e}", file=sys.stderr)
            return 1

        version = store.set_password(user.user_id, new_hash)

        print(f"\nPassword reset for {user.email}.")
        if args.generate:
            print(f"  New password: {password}")
            print("  Shown once — copy it now.")
        print(f"  All existing sessions are signed out (token version {version}).")
        return 0

    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
