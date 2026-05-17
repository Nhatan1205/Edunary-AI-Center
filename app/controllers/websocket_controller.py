"""
WebSocket controller — WebSocket endpoint for real-time chat.

Auth flow (token-in-first-message):
  1. Client connects to /ws/{session_id}  ← no token in URL, won't appear in access logs
  2. If authenticated, client sends first message: {"type": "auth", "token": "<jwt>"}
     Controller verifies the JWT and extracts user_id — silently falls back to anonymous
     if token is absent or invalid.
  3. All subsequent messages are treated as normal chat messages.
"""

import json
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.chat_service import process_message
from app.services.websocket_manager import manager, stream_response
from app.guardrails.input_guard import sanitize_input, check_input_length, check_prompt_injection
from app.guardrails.output_guard import check_output_safety, get_fallback_reply
from app.guardrails.rate_limiter import RateLimiter
from app.auth.jwt_handler import verify_token
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

_rate_limiter = RateLimiter(max_requests=15, window_seconds=60)


def _extract_user_id_from_token(token: Optional[str]) -> Optional[str]:
    """
    Verify a JWT token and extract the ASP.NET Identity user ID.

    Returns None (not an error) if the token is absent, invalid, or expired.
    The chat flow degrades gracefully to anonymous mode.
    Token value is NEVER logged.
    """
    if not token:
        return None
    try:
        payload = verify_token(token)
        # ASP.NET Identity stores the user ID under 'nameid' or 'sub'
        user_id = payload.get("nameid") or payload.get("sub") or payload.get("UserId")
        if user_id:
            logger.info(f"WebSocket: authenticated user_id={user_id}")
        return user_id
    except Exception as e:
        logger.warning(f"WebSocket: JWT verification failed ({e}) — falling back to anonymous.")
        return None


@router.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat.

    Protocol:
      - Client connects (no token in URL).
      - Optional first message: {"type": "auth", "token": "<jwt>"}
        → verifies identity, enables personalisation.
      - All other messages: {"message": "<text>", "session_id": "<id>"}
        → processed as chat.

    Args:
        websocket: WebSocket connection.
        session_id: Unique session identifier from client.
    """
    await manager.connect(websocket, session_id)

    # user_id is resolved once per connection via the auth handshake message
    user_id: Optional[str] = None

    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)

            # ── Auth handshake ──────────────────────────────────────────────
            if message_data.get("type") == "auth":
                token = message_data.get("token")
                user_id = _extract_user_id_from_token(token)
                # No reply needed — the client continues after sending auth
                continue

            # ── Normal chat message ─────────────────────────────────────────
            user_message = message_data.get("message", "")
            if not user_message:
                continue

            # ── Rate limit ─────────────────────────────────────────────────
            if not _rate_limiter.allow(session_id):
                await manager.send_message(
                    json.dumps({"type": "error", "content": "Too many messages. Please slow down."}),
                    session_id,
                )
                continue

            # ── Input guardrails ───────────────────────────────────────────
            user_message = sanitize_input(user_message)

            if check_input_length(user_message):
                await manager.send_message(
                    json.dumps({"type": "error", "content": "Message too long. Please keep it under 2000 characters."}),
                    session_id,
                )
                continue

            if check_prompt_injection(user_message):
                await stream_response(
                    "I can only help with Edunary courses and platform features. How can I assist you?",
                    session_id,
                )
                continue

            # ── Process ────────────────────────────────────────────────────
            result = process_message(session_id, user_message, user_id=user_id)

            # ── Output guardrail ───────────────────────────────────────────
            is_safe, _ = check_output_safety(result["reply"])
            if not is_safe:
                result["reply"] = get_fallback_reply()

            # ── Stream response ────────────────────────────────────────────
            await stream_response(result["reply"], session_id)

            if result.get("messageType") == "course_cards" and result.get("courses"):
                await manager.send_message(json.dumps({
                    "type": "course_cards",
                    "courses": result["courses"],
                    "is_personalized": result.get("is_personalized", False),
                }), session_id)

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(session_id)
