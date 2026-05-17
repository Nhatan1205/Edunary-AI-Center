"""
Output guardrail — validates LLM responses before returning them to the user.

Checks:
  - Empty / blank response
  - LLM identity leak phrases (model referencing itself inappropriately)
"""

from app.core.logging import get_logger

logger = get_logger(__name__)

# Phrases that indicate the LLM leaked its identity or is in a bad state
_LEAK_PHRASES = [
    "as an ai language model",
    "as a large language model",
    "i am an ai",
    "i'm an ai",
]

_FALLBACK_REPLY = (
    "I'm having trouble generating a response right now. Please try again."
)


def check_output_safety(text: str) -> tuple[bool, str]:
    """
    Validate LLM output before sending to the user.

    Returns:
        (is_safe: bool, reason: str)
    """
    if not text or not text.strip():
        logger.warning("[OutputGuard] Blocked: empty response")
        return False, "empty_response"

    lower = text.lower()
    for phrase in _LEAK_PHRASES:
        if phrase in lower:
            logger.warning(f"[OutputGuard] Blocked: identity leak — '{phrase}'")
            return False, "identity_leak"

    return True, ""


def get_fallback_reply() -> str:
    """Return the standard fallback message for unsafe outputs."""
    return _FALLBACK_REPLY
