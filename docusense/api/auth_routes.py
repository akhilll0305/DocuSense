"""
Authentication endpoints.

POST /api/auth/register  - create an account and return a token
POST /api/auth/login     - exchange credentials for a token
GET  /api/auth/me        - the authenticated user

Author: DocuSense
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel, EmailStr, Field

from docusense.api.deps import get_current_user, get_user_store
from docusense.auth import (
    AuthError,
    DuplicateEmailError,
    User,
    UserStore,
    create_access_token,
    hash_password,
    verify_password,
)
from docusense.config.settings import settings


router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# ==============================================================================
# SCHEMAS
# ==============================================================================

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)
    name: str = Field(default="", max_length=100)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=72)


class UserResponse(BaseModel):
    user_id: str
    email: str
    name: str
    created_at: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


def _token_response(user: User) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(user.user_id, user.email),
        expires_in=settings.access_token_expire_minutes * 60,
        user=UserResponse(**user.public()),
    )


# ==============================================================================
# ROUTES
# ==============================================================================

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest, store: UserStore = Depends(get_user_store)):
    """Create an account. Returns a token so the client is logged straight in."""
    if len(request.password) < settings.min_password_length:
        raise HTTPException(
            status_code=422,
            detail=f"Password must be at least {settings.min_password_length} characters",
        )

    try:
        user = store.create_user(
            email=request.email,
            name=request.name,
            password_hash=hash_password(request.password),
        )
    except DuplicateEmailError:
        raise HTTPException(status_code=409, detail="Email already registered") from None
    except AuthError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None

    logger.info(f"Registered {user.user_id}")
    return _token_response(user)


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, store: UserStore = Depends(get_user_store)):
    """Exchange email and password for an access token."""
    user = store.get_by_email(request.email)

    # Same response whether the email is unknown or the password is wrong, so
    # the endpoint cannot be used to enumerate registered addresses.
    if user is None or not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.info(f"Login {user.user_id}")
    return _token_response(user)


@router.get("/me", response_model=UserResponse)
async def me(user: User = Depends(get_current_user)):
    """Return the authenticated user."""
    return UserResponse(**user.public())
