"""
Authentication for DocuSense.

Components:
-----------
1. security: bcrypt password hashing, JWT issuing and verification
2. store: SQLite-backed user accounts
"""

from .security import (
    AuthError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from .store import DuplicateEmailError, User, UserStore

__all__ = [
    "AuthError",
    "DuplicateEmailError",
    "User",
    "UserStore",
    "create_access_token",
    "decode_access_token",
    "hash_password",
    "verify_password",
]
