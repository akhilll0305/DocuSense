"""
Authentication endpoints.

POST /api/auth/register    - create an account and return a token
POST /api/auth/login       - exchange credentials for a token
GET  /api/auth/me          - the authenticated user
POST /api/auth/logout      - revoke the presented token
POST /api/auth/logout-all  - revoke every token for this account
POST /api/auth/password    - change the password and re-issue a token

Author: DocuSense
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel, EmailStr, Field

from docusense.api.deps import get_current_user, get_token_claims, get_user_store
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


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=72)
    new_password: str = Field(min_length=8, max_length=72)


class MessageResponse(BaseModel):
    message: str
    success: bool = True


def _token_response(user: User, token_version: int | None = None) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(
            user.user_id,
            user.email,
            token_version=token_version if token_version is not None else user.token_version,
        ),
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


@router.post("/logout", response_model=MessageResponse)
async def logout(
    user: User = Depends(get_current_user),
    claims: dict = Depends(get_token_claims),
    store: UserStore = Depends(get_user_store),
):
    """
    Sign out this session.

    The presented token is recorded as revoked and stops working immediately;
    the user's other devices are untouched. Clearing the client's copy alone,
    which is all logout used to do, leaves a working token in whatever else
    has it — a browser history entry, a proxy log, a copied curl command.
    """
    jti = claims.get("jti")
    if jti:
        store.revoke_token(jti, user.user_id, int(claims.get("exp", 0)))
        logger.info(f"Signed out session {jti[:8]}… for {user.user_id}")
        return MessageResponse(message="Signed out")

    # Tokens minted before revocation existed carry no id, so there is nothing
    # to record. Say so rather than reporting a revocation that did not happen.
    logger.info(f"Logout for {user.user_id} on a token with no id; nothing revoked")
    return MessageResponse(
        message="Signed out on this device. This token predates session "
                "revocation and stays valid until it expires; use "
                "/api/auth/logout-all to invalidate it now.",
        success=True,
    )


@router.post("/logout-all", response_model=MessageResponse)
async def logout_all(
    user: User = Depends(get_current_user),
    store: UserStore = Depends(get_user_store),
):
    """
    Sign out every session for this account, including this one.

    One write, so it covers tokens this server has no record of.
    """
    version = store.bump_token_version(user.user_id)
    return MessageResponse(
        message=f"Signed out of all sessions (token version {version})"
    )


@router.post("/password", response_model=TokenResponse)
async def change_password(
    request: ChangePasswordRequest,
    user: User = Depends(get_current_user),
    store: UserStore = Depends(get_user_store),
):
    """
    Change the password, given the current one.

    Every existing session is invalidated, including the caller's: a password
    change that leaves the old sessions running does not lock anyone out. A
    fresh token comes back so the caller stays signed in without a round trip
    through the login form.
    """
    if not verify_password(request.current_password, user.password_hash):
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    if request.new_password == request.current_password:
        raise HTTPException(
            status_code=422, detail="New password must differ from the current one"
        )

    if len(request.new_password) < settings.min_password_length:
        raise HTTPException(
            status_code=422,
            detail=f"Password must be at least {settings.min_password_length} characters",
        )

    try:
        new_hash = hash_password(request.new_password)
    except AuthError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None

    version = store.set_password(user.user_id, new_hash)
    return _token_response(user, token_version=version)
