"""
Chat controller — REST API endpoint for chat.
Replaces routes/chat.py.
"""

import uuid

from fastapi import APIRouter, HTTPException, Depends

from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.services.chat_service import process_message
from app.auth.dependencies import get_current_user
from app.guardrails.input_guard import sanitize_input, check_input_length, check_prompt_injection
from app.guardrails.output_guard import check_output_safety, get_fallback_reply
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest, user=Depends(get_current_user)):
    """
    Process a chat message and return a response.

    - Generates a session_id if not provided.
    - Classifies intent and dispatches to the appropriate handler.
    - Returns text reply and optional course cards.
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())

        # ── Input guardrails ───────────────────────────────────────────
        message = sanitize_input(request.message)

        if check_input_length(message):
            return ChatResponse(
                session_id=session_id,
                reply="Please keep your message under 2000 characters.",
                messageType="text",
            )

        if check_prompt_injection(message):
            return ChatResponse(
                session_id=session_id,
                reply="I can only help with Edunary courses and platform features. How can I assist you?",
                messageType="text",
            )

        # ── Process ────────────────────────────────────────────────────
        result = process_message(session_id, message)

        # ── Output guardrail ───────────────────────────────────────────
        is_safe, _ = check_output_safety(result["reply"])
        if not is_safe:
            result["reply"] = get_fallback_reply()

        return ChatResponse(
            session_id=session_id,
            reply=result["reply"],
            messageType=result["messageType"],
            courses=result.get("courses"),
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
