"""
Chat service — main orchestration layer for chat messages.

Flow: get history → load user profile (if authenticated) → classify intent
      → dispatch to handler → save history → return result.

Supports both anonymous (no user_id) and personalized (user_id provided) modes.
"""

from typing import Optional

from app.services.intent_service import classify_intent
from app.services.course_service import handle_course_search, handle_personalized_recommendation
from app.services.faq_service import handle_faq
from app.services.user_profile_service import get_user_profile, UserProfileContext
from app.store.chat_history_store import get_session_history, save_message
from app.core.logging import get_logger, timer

logger = get_logger(__name__)


def _format_history(history_data: list[dict]) -> str:
    lines = []
    for msg in history_data:
        role = msg["type"]
        role_name = "user" if role == "human" else "assistant"
        lines.append(f"{role_name}: {msg['data']['content']}")
    return "\n".join(lines)


def _handle_greeting(message: str, user_context: Optional[UserProfileContext]) -> dict:
    if user_context:
        reply = f"Hi {user_context.display_name}! 👋 How can I help you today?"
    else:
        reply = "Hi! Nice to meet you 👋 How can I help you find the perfect course today?"
    return {
        "reply": reply,
        "messageType": "text",
        "courses": None,
        "is_personalized": user_context is not None,
    }


def _handle_fallback(user_context: Optional[UserProfileContext]) -> dict:
    if user_context:
        reply = f"Sorry {user_context.display_name}, I can only help with Edunary courses and platform questions. Is there a topic you'd like to explore?"
    else:
        reply = "Sorry, I can only answer questions about Edunary courses and platform features. How can I help you today?"
    return {
        "reply": reply,
        "messageType": "text",
        "courses": None,
        "is_personalized": user_context is not None,
    }


def process_message(
    session_id: str,
    message: str,
    user_id: Optional[str] = None,
) -> dict:
    """
    Process a chat message and return a response.

    Args:
        session_id: Unique session identifier (used for chat history).
        message: The user's message text.
        user_id: ASP.NET Identity GUID. When provided, the response is personalized
                 using the user's Qdrant-embedded profile. When None, falls back to
                 the anonymous chat flow.

    Returns:
        dict with keys: reply, messageType, courses, is_personalized
    """
    logger.info(f">>> START: session={session_id} | user_id={user_id or 'anonymous'}")

    # Load conversation history
    history_data = get_session_history(session_id)
    recent_history = history_data[-10:]
    history_text = _format_history(recent_history)

    # Load user profile (returns None if user is anonymous or not yet embedded)
    user_context: Optional[UserProfileContext] = None
    if user_id:
        with timer("User profile lookup"):
            user_context = get_user_profile(user_id)

    # Classify intent (passes user_context for personalized intent detection)
    intent = classify_intent(message, history_text, user_context)

    with timer("Handler dispatch"):
        if "greeting" in intent:
            result = _handle_greeting(message, user_context)
        elif "personalized_recommendation" in intent and user_context:
            result = handle_personalized_recommendation(user_context, history_text)
        elif "search_faq" in intent:
            result = handle_faq(message)
            result["is_personalized"] = user_context is not None
        elif "get_course_information" in intent:
            result = handle_course_search(message, history_text, user_context)
        else:
            result = _handle_fallback(user_context)

    save_message(session_id, "user", message)
    save_message(session_id, "assistant", result["reply"])

    logger.info(
        f"<<< DONE: intent={intent} | messageType={result['messageType']} | "
        f"personalized={result.get('is_personalized', False)}"
    )
    return result
