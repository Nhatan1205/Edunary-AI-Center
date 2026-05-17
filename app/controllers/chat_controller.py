"""
Chat controller — REST API endpoint for chat.
Replaces routes/chat.py.
"""

import uuid

from fastapi import APIRouter, HTTPException, Depends

from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.services.chat_service import process_message
from app.auth.dependencies import get_current_user
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest, user=Depends(get_current_user)):
    """
    Process a chat message and return a response.

    - Generates a session_id if not provided.
    - Extracts user_id from the authenticated user (JWT) when available;
      falls back to None for anonymous requests.
    - Classifies intent and dispatches to the appropriate handler.
    - Returns text reply and optional course cards.
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())

        # Extract user_id from JWT payload if authenticated
        user_id: str | None = None
        if user and isinstance(user, dict):
            # JWT claim for ASP.NET Identity user ID is typically 'sub' or 'nameid'
            user_id = user.get("sub") or user.get("nameid") or user.get("UserId")

        result = process_message(session_id, request.message, user_id=user_id)

        return ChatResponse(
            session_id=session_id,
            reply=result["reply"],
            messageType=result["messageType"],
            courses=result.get("courses"),
            is_personalized=result.get("is_personalized", False),
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
