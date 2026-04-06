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
        token: The JWT token string.

    Returns:
        dict: Token payload (userId, email, roles, etc.)

    Raises:
        AuthenticationError: If the token is invalid or expired.

    TODO: Implement when JWT secret key and algorithm are confirmed.
    """
    settings = get_settings()

    if not settings.JWT_SECRET_KEY:
        logger.warning("JWT_SECRET_KEY not configured — cannot verify tokens")
        raise AuthenticationError("Authentication not configured")

    try:
        from jose import jwt, JWTError

        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return payload
    except Exception:
        raise AuthenticationError("Invalid or expired token")
