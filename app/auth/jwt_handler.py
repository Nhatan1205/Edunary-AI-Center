"""
JWT token verification.
Placeholder — will be implemented when auth requirements are finalized.
"""

from app.core.config import get_settings
from app.core.exceptions import AuthenticationError
from app.core.logging import get_logger

logger = get_logger(__name__)


def verify_token(token: str) -> dict:
    """
    Verify a JWT token issued by the .NET Backend.

    Args:
        token: The JWT token string (without 'Bearer ' prefix).

    Returns:
        dict: Decoded token payload (sub/nameid, email, roles, etc.)

    Raises:
        AuthenticationError: If the token is invalid, expired, or key not configured.
    """
    settings = get_settings()

    secret_key = settings.ACCESS_TOKEN_KEY
    if not secret_key:
        logger.warning("ACCESS_TOKEN_KEY not configured — cannot verify JWT tokens")
        raise AuthenticationError("Authentication not configured")

    try:
        from jose import jwt, JWTError

        payload = jwt.decode(
            token,
            secret_key,
            algorithms=[settings.JWT_ALGORITHM],
            options={"verify_aud": False},  # .NET doesn't set audience by default
        )
        return payload
    except Exception as e:
        raise AuthenticationError(f"Invalid or expired token: {e}")
