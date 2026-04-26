"""
Static API Key authentication for service-to-service calls.

The .NET backend sends the key in the X-API-Key header on every request.
This dependency is applied at the router level — all routes under a router
that uses this will require a valid API key.

Usage in a controller:
    router = APIRouter(dependencies=[Depends(verify_api_key)])
"""

from fastapi import Header
from app.core.config import get_settings
from app.core.exceptions import AuthenticationError


async def verify_api_key(x_api_key: str = Header(None)) -> bool:
    """
    Verify the X-API-Key header sent by the .NET backend.

    - When AUTH_ENABLED=false (dev mode): skip verification — allow all.
    - When AUTH_ENABLED=true (production): require a valid key matching AI_CENTER_API_KEY.

    Raises:
        AuthenticationError: If the key is missing or invalid.
    """
    settings = get_settings()

    if not settings.AUTH_ENABLED:
        return True  # Dev mode — skip auth

    if not x_api_key:
        raise AuthenticationError("X-API-Key header is missing")

    if x_api_key != settings.AI_CENTER_API_KEY:
        raise AuthenticationError("Invalid API key")

    return True
