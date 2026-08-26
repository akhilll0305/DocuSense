"""
Shared FastAPI dependencies.

Holds the process-wide RAG instance and user store, and resolves the bearer
token on every protected request.

Author: DocuSense
"""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from docusense.auth import AuthError, User, UserStore, decode_access_token

# Populated during application startup (see api/app.py)
_rag_instance = None
_user_store: Optional[UserStore] = None

# auto_error=False so a missing header yields our 401 with a WWW-Authenticate
# challenge rather than FastAPI's bare 403.
_bearer = HTTPBearer(auto_error=False)


def set_rag_instance(rag) -> None:
    """Register the process-wide RAG instance."""
    global _rag_instance
    _rag_instance = rag


def set_user_store(store: UserStore) -> None:
    """Register the process-wide user store."""
    global _user_store
    _user_store = store


def get_rag():
    """Dependency: the RAG instance."""
    if _rag_instance is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    return _rag_instance


def get_user_store() -> UserStore:
    """Dependency: the user store."""
    if _user_store is None:
        raise HTTPException(status_code=503, detail="User store not initialized")
    return _user_store


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    store: UserStore = Depends(get_user_store),
) -> User:
    """
    Resolve the authenticated user from the Authorization header.

    Raises:
        HTTPException 401: when the header is missing, the token is invalid or
            expired, or the subject no longer exists
    """
    if credentials is None or not credentials.credentials:
        raise _unauthorized("Not authenticated")

    try:
        claims = decode_access_token(credentials.credentials)
    except AuthError as e:
        raise _unauthorized(str(e)) from None

    user_id = claims.get("sub")
    if not user_id:
        raise _unauthorized("Invalid authentication token")

    user = store.get_by_id(user_id)
    if user is None:
        # Token signature was valid but the account is gone.
        raise _unauthorized("User no longer exists")

    return user
