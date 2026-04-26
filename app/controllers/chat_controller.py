"""
Chat controller — REST API endpoint for chat.
Replaces routes/chat.py.
"""

import uuid

from fastapi import APIRouter, HTTPException, Depends

from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.services.chat_service import process_message
from app.auth.dependencies import get_current_user
from app.auth.api_key import verify_api_key
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(verify_api_key)])


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
        result = process_message(session_id, request.message)

        return ChatResponse(
            session_id=session_id,
            reply=result["reply"],
            messageType=result["messageType"],
            courses=result.get("courses"),
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
