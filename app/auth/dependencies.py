"""
Authentication dependencies for FastAPI routes.
"""
import hashlib
import hmac
import time
from typing import Optional

from lark import logger

from fastapi import Header,Request

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


"""
HMAC-SHA256 request verification for service-to-service auth.

.NET backend signs each request with:
  body_hash  = SHA256(raw_body)
  message    = "{timestamp}:{body_hash}"
  signature  = HMAC-SHA256(secret, message)

Headers expected:
  X-Timestamp: unix seconds (string)
  X-Signature: hex string
"""
TOLERANCE_SECONDS = 300
async def verify_request(request: Request, x_timestamp: str = Header(None), x_signature: str = Header(None)) -> bool:
    settings = get_settings()

    if not settings.AUTH_ENABLED:
        return True  # skip auth

    # 1. Validate headers present
    if not x_timestamp or not x_signature:
        raise AuthenticationError("Missing X-Timestamp or X-Signature header")

    # 2. Reject stale/replayed requests
    try:
        req_time = int(x_timestamp)
    except ValueError:
        raise AuthenticationError("Invalid X-Timestamp")

    if abs(time.time() - req_time) > TOLERANCE_SECONDS:
        raise AuthenticationError("Request timestamp expired")

    # 3. Re-compute expected signature
    body = await request.body()
    body_hash = hashlib.sha256(body).hexdigest()
    message = f"{x_timestamp}:{body_hash}"

    secret = settings.AI_CENTER_API_KEY.encode()
    expected = hmac.new(secret, message.encode(), hashlib.sha256).hexdigest()

    print(f"Body hash: {body_hash}")
    print(f"Message: {message}")
    print(f"Expected: {expected}")
    print(f"Received: {x_signature}")

    # 4. Constant-time compare
    if not hmac.compare_digest(expected, x_signature.lower()):
        raise AuthenticationError("Invalid signature")

    return True
