"""
Input guardrail — validates user messages before they reach the LLM.

Checks:
  - Input length (token abuse prevention)
  - Unicode sanitization (zero-width / control char smuggling)
  - Prompt injection patterns (jailbreak attempts)
"""

import re

from app.core.logging import get_logger

logger = get_logger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

MAX_INPUT_LENGTH = 2000

# Regex patterns that indicate prompt injection / jailbreak attempts
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|rules?|prompts?|guidelines?)",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"pretend\s+(you\s+are|to\s+be)",
    r"(forget|disregard|override)\s+(your|all)\s+(instructions?|rules?|guidelines?|training)",
    r"act\s+as\s+(if|though)",
    r"(jailbreak|dan\s+mode|developer\s+mode)",
    r"\bsystem\s*:\s*",        # Fake system message injection
    r"\bdo\s+anything\s+now\b",  # DAN prompt pattern
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

# Zero-width and invisible unicode characters used for smuggling
_UNICODE_SMUGGLING_RE = re.compile(
    r"[\u200b-\u200f\u2028-\u202f\u2060-\u206f\ufeff]"
)


# ─── Public API ───────────────────────────────────────────────────────────────

def sanitize_input(text: str) -> str:
    """Strip zero-width / invisible unicode characters, then trim whitespace."""
    return _UNICODE_SMUGGLING_RE.sub("", text).strip()


def check_input_length(text: str) -> bool:
    """Return True if input exceeds MAX_INPUT_LENGTH."""
    return len(text) > MAX_INPUT_LENGTH


def check_prompt_injection(text: str) -> bool:
    """Return True if text matches any known prompt injection pattern."""
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(text):
            logger.warning("[InputGuard] Prompt injection blocked")
            return True
    return False
