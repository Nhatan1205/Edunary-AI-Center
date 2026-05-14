"""
WebSocket controller — WebSocket endpoint for real-time chat.
Replaces routes/websocket_chat.py.
No authentication on WebSocket (as requested).
"""

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.chat_service import process_message
from app.services.websocket_manager import manager, stream_response
from app.guardrails.input_guard import sanitize_input, check_input_length, check_prompt_injection
from app.guardrails.output_guard import check_output_safety, get_fallback_reply
from app.guardrails.rate_limiter import RateLimiter
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

_rate_limiter = RateLimiter(max_requests=15, window_seconds=60)


@router.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat.

    Args:
        websocket: WebSocket connection.
        session_id: Unique session identifier from client.
    """
    await manager.connect(websocket, session_id)

    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
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
            result = process_message(session_id, user_message)

            # ── Output guardrail ───────────────────────────────────────────
            is_safe, _ = check_output_safety(result["reply"])
            if not is_safe:
                result["reply"] = get_fallback_reply()

            await stream_response(result["reply"], session_id)

            if result.get("messageType") == "course_cards" and result.get("courses"):
                await manager.send_message(json.dumps({
                    "type": "course_cards",
                    "courses": result["courses"]
                }), session_id)

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(session_id)
