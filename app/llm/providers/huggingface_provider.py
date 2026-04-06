"""HuggingFace Router LLM provider (OpenAI-compatible API)."""

from langchain_openai import ChatOpenAI
from app.core.config import Settings


def create_huggingface(settings: Settings):
    """Create a ChatOpenAI instance pointed at HuggingFace Router."""
    if not settings.HF_TOKEN:
        raise ValueError("HF_TOKEN not found in environment variables")

    return ChatOpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=settings.HF_TOKEN,
        model=settings.HUGGINGFACE_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )
