"""
Intent classification service.
Determines the user's intent from their message and conversation history.
"""

from app.prompts.chat_prompts import INTENT_CLASSIFIER_PROMPT
from app.llm.factory import create_llm
from app.core.logging import get_logger

logger = get_logger(__name__)


def classify_intent(message: str, history_text: str) -> str:
    """
    Classify user intent using LLM.

    Args:
        message: The user's latest message.
        history_text: Formatted conversation history string.

    Returns:
        str: One of 'greeting', 'get_course_information', 'search_faq', 'fallback'.
    """
    llm = create_llm()

    intent_result = (INTENT_CLASSIFIER_PROMPT | llm).invoke({
        "chat_history": history_text,
        "input": message,
    })

    raw = intent_result.content if hasattr(intent_result, "content") else intent_result
    intent = raw.strip().lower()

    logger.info(f"Intent: '{intent}'")
    return intent
