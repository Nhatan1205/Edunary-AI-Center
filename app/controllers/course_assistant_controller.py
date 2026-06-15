"""
Course Assistant Controller — REST API endpoints.
Protected by verify_request (HMAC from .NET backend).

Endpoints:
  POST /api/course-assistant/chat      — process a student message
  POST /api/course-assistant/history   — get paginated conversation history
  POST /api/course-assistant/clear     — clear conversation
"""

from fastapi import APIRouter, Depends, HTTPException

from app.auth.dependencies import verify_request
from app.core.logging import get_logger
from app.schemas.course_assistant_schema import (
    CourseAssistantChatRequest,
    CourseAssistantChatResponse,
    CourseAssistantHistoryRequest,
    CourseAssistantHistoryResponse,
    CourseAssistantClearRequest,
)
from app.services.course_assistant_service import process_assistant_message
from app.store.course_assistant_store import (
    get_or_create_conversation,
    get_messages_paginated,
    clear_conversation,
)

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(verify_request)])


@router.post("/chat", response_model=CourseAssistantChatResponse)
async def chat(request: CourseAssistantChatRequest):
    """
    Process a student's message and return the AI assistant's reply.

    The .NET backend is responsible for:
    - Verifying the student's JWT
    - Checking enrollment before forwarding here

    This endpoint trusts the user_id provided in the request body.
    """
    logger.info(
        f"[CourseAssistant] Chat request: user={request.user_id} "
        f"course={request.course_id} content_type={request.content_type}"
    )

    try:
        result = process_assistant_message(
            user_id=request.user_id,
            course_id=request.course_id,
            course_data=request.course_data,
            content_id=request.content_id,
            content_type=request.content_type,
            media_type=request.media_type,
            content_title=request.content_title,
            message=request.message,
            llm_config=request.llm_config,
        )
        return CourseAssistantChatResponse(
            reply=result["reply"],
            message_type=result.get("message_type", "text"),
            sources=result.get("sources"),
        )

    except Exception as e:
        logger.error(f"[CourseAssistant] Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal AI error")


@router.post("/history", response_model=CourseAssistantHistoryResponse)
async def get_history(request: CourseAssistantHistoryRequest):
    """
    Get paginated conversation history for a student's course.
    Returns newest messages first. Use cursor for older pages.
    """
    logger.info(
        f"[CourseAssistant] History request: user={request.user_id} "
        f"course={request.course_id} cursor={request.cursor}"
    )

    try:
        conversation_id = get_or_create_conversation(
            user_id=request.user_id,
            course_id=request.course_id,
        )
        result = get_messages_paginated(
            conversation_id=conversation_id,
            cursor=request.cursor,
            page_size=request.page_size,
        )
        return CourseAssistantHistoryResponse(
            items=result["items"],
            has_more=result["has_more"],
        )

    except Exception as e:
        logger.error(f"[CourseAssistant] History error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve history")


@router.post("/clear")
async def clear_history(request: CourseAssistantClearRequest):
    """Clear all conversation history for a student's course."""
    logger.info(
        f"[CourseAssistant] Clear request: user={request.user_id} course={request.course_id}"
    )

    try:
        clear_conversation(user_id=request.user_id, course_id=request.course_id)
        return {"success": True, "message": "Conversation cleared."}

    except Exception as e:
        logger.error(f"[CourseAssistant] Clear error: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear conversation")
