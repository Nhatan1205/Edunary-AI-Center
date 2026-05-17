"""
Intent classification service.
Determines the user's intent from their message and conversation history.
Supports an optional user profile context for personalized intent detection.
"""

from typing import TYPE_CHECKING

from app.prompts.chat_prompts import INTENT_CLASSIFIER_PROMPT
from app.llm.factory import create_llm
from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.services.user_profile_service import UserProfileContext

logger = get_logger(__name__)


def classify_intent(
    message: str,
    history_text: str,
    user_context: "UserProfileContext | None" = None,
) -> str:
    """
    Classify user intent using LLM.

    Args:
        message: The user's latest message.
        history_text: Formatted conversation history string.
        user_context: Optional user profile for personalized intent detection.

    Returns:
        str: One of 'greeting', 'get_course_information', 'personalized_recommendation',
             'search_faq', 'fallback'.
    """
    llm = create_llm()

    user_context_text = user_context.to_prompt_text() if user_context else "(not logged in)"

    intent_result = (INTENT_CLASSIFIER_PROMPT | llm).invoke({
        "chat_history": history_text,
        "input": message,
        "user_context": user_context_text,
    })

    raw = intent_result.content if hasattr(intent_result, "content") else intent_result
    intent = raw.strip().lower()

    # Fallback: if LLM returns 'personalized_recommendation' but there's no user context,
    # treat it as 'get_course_information' to avoid a crash downstream.
    if intent == "personalized_recommendation" and user_context is None:
        logger.warning("LLM returned 'personalized_recommendation' but no user context — downgrading to 'get_course_information'.")
        intent = "get_course_information"

    logger.info(f"Intent: '{intent}' | personalized={user_context is not None}")
    return intent
