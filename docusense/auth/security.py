"""
Password hashing and JWT token handling.

Passwords are hashed with bcrypt (never stored or logged in plaintext).
Sessions are stateless JWTs signed with the application secret.

Author: DocuSense
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import bcrypt
import jwt
from jwt import ExpiredSignatureError, InvalidTokenError
from loguru import logger

from docusense.config.settings import settings


# bcrypt truncates at 72 bytes and raises on longer input in 4.x+.
MAX_PASSWORD_BYTES = 72


class AuthError(Exception):
    """Raised when a credential or token is invalid."""


# ==============================================================================
# PASSWORDS
# ==============================================================================

def hash_password(password: str) -> str:
    """
    Hash a plaintext password with bcrypt.

    Args:
        password: Plaintext password

    Returns:
        The bcrypt hash, as a UTF-8 string safe to store

    Raises:
        AuthError: if the password is empty or too long for bcrypt
    """
    if not password:
        raise AuthError("Password must not be empty")

    encoded = password.encode("utf-8")
    if len(encoded) > MAX_PASSWORD_BYTES:
        raise AuthError(
            f"Password must be at most {MAX_PASSWORD_BYTES} bytes "
            f"({len(encoded)} given)"
        )

    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """
    Check a plaintext password against a stored bcrypt hash.

    Returns False rather than raising on malformed input, so callers can treat
    every failure path identically and avoid leaking which part was wrong.
    """
    if not password or not password_hash:
        return False

    encoded = password.encode("utf-8")
    if len(encoded) > MAX_PASSWORD_BYTES:
        return False

    try:
        return bcrypt.checkpw(encoded, password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ==============================================================================
# TOKENS
# ==============================================================================

def create_access_token(
    user_id: str,
    email: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Issue a signed JWT for a user.

    Args:
        user_id: Stable user identifier (the token subject)
        email: User's email, carried for convenience
        expires_delta: Lifetime override; defaults to settings

    Returns:
        Encoded JWT
    """
    expire_minutes = settings.access_token_expire_minutes
    expires = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=expire_minutes)
    )

    payload: Dict[str, Any] = {
        "sub": user_id,
        "email": email,
        "exp": expires,
        "iat": datetime.now(timezone.utc),
    }

    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode a JWT.

    Args:
        token: Encoded JWT

    Returns:
        The decoded claims

    Raises:
        AuthError: if the token is expired, malformed, or badly signed
    """
    try:
        return jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except ExpiredSignatureError:
        raise AuthError("Token has expired") from None
    except InvalidTokenError as e:
        # Do not echo the token or the library's detail back to the client.
        logger.debug(f"Rejected token: {e}")
        raise AuthError("Invalid authentication token") from None
