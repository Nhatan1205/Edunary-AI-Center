"""
Chat service — main orchestration layer for chat messages.
Refactored from services/chat_services.py.

Flow: get history → classify intent → dispatch to handler → save history → return result.
"""

from app.services.intent_service import classify_intent
from app.services.course_service import handle_course_search
from app.services.faq_service import handle_faq
from app.store.chat_history_store import get_session_history, save_message
from app.core.logging import get_logger

logger = get_logger(__name__)


def _format_history(history_data: list[dict]) -> str:
    """
    Format raw history dicts into a readable string for prompts.

    Args:
        history_data: List of message dicts from MongoDB.

    Returns:
        str: Formatted history string.
    """
    lines = []
    for msg in history_data:
        role = msg["type"]
        role_name = "user" if role == "human" else "assistant"
        lines.append(f"{role_name}: {msg['data']['content']}")
    return "\n".join(lines)


def _handle_greeting(message: str) -> dict:
    """Handle greeting intent."""
    return {
        "reply": "Hi! Nice to meet you 👋 How can I help you find the perfect course today?",
        "messageType": "text",
        "courses": None,
    }


def _handle_fallback() -> dict:
    """Handle fallback/unknown intent."""
    return {
        "reply": "Sorry, I can only answer questions about Edunary courses and platform features. How can I help you today?",
        "messageType": "text",
        "courses": None,
    }


def process_message(session_id: str, message: str) -> dict:
    """
    Process a chat message end-to-end.

    1. Load conversation history
    2. Classify intent via LLM
    3. Dispatch to appropriate handler
    4. Save messages to history
    5. Return response dict

    Args:
        session_id: Chat session identifier.
        message: User's message text.

    Returns:
        dict with keys: reply (str), messageType (str), courses (list|None)
    """
    logger.info(f"Processing message: session={session_id}, message='{message[:80]}...'")

    # Build history text from last 10 messages
    history_data = get_session_history(session_id)
    recent_history = history_data[-10:]
    history_text = _format_history(recent_history)

    # Classify intent
    intent = classify_intent(message, history_text)

    # Dispatch to handler
    if "greeting" in intent:
        result = _handle_greeting(message)
    elif "search_faq" in intent:
        result = handle_faq(message)
    elif "get_course_information" in intent:
        result = handle_course_search(message, history_text)
    else:
        result = _handle_fallback()

    # Persist conversation
    save_message(session_id, "user", message)
    save_message(session_id, "assistant", result["reply"])

    logger.info(f"Response: intent={intent}, messageType={result['messageType']}")
    return result
