"""
Authentication dependencies for FastAPI routes.
"""

from typing import Optional

from fastapi import Header

from app.core.config import get_settings
from app.core.exceptions import AuthenticationError
from app.auth.jwt_handler import verify_token


async def get_current_user(authorization: str = Header(None)) -> Optional[dict]:
    """
    FastAPI dependency — extracts the current user from the Authorization header.

    When AUTH_ENABLED=false (development), this returns None (no auth required).
    When AUTH_ENABLED=true (production), a valid JWT Bearer token is required.

    Usage in controllers:
        @router.post("/")
        def my_endpoint(user=Depends(get_current_user)):
            ...
    """
    settings = get_settings()

    if not settings.AUTH_ENABLED:
        return None  # Auth disabled — skip

    if not authorization:
        raise AuthenticationError("Authorization header is missing")

    # Extract token from "Bearer <token>"
    if not authorization.startswith("Bearer "):
        raise AuthenticationError("Invalid authorization header format. Use: Bearer <token>")

    token = authorization[7:]  # Remove "Bearer " prefix
    return verify_token(token)
