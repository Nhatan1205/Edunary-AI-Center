"""Custom OpenAI-compatible API provider.

Use this to call any API that follows OpenAI's chat completions format
(e.g. self-hosted vLLM, LiteLLM, LocalAI, etc.).

Set in .env:
  CUSTOM_LLM_BASE_URL=https://your-api-host/v1
  CUSTOM_LLM_API_KEY=your_key
  CUSTOM_LLM_MODEL=your_model_name
"""

from langchain_openai import ChatOpenAI
from app.core.config import Settings


def create_custom(settings: Settings):
    """Create a ChatOpenAI instance pointed at a custom base URL."""
    if not settings.CUSTOM_LLM_BASE_URL:
        raise ValueError("CUSTOM_LLM_BASE_URL not found in environment variables")

    return ChatOpenAI(
        base_url=settings.CUSTOM_LLM_BASE_URL,
        api_key=settings.CUSTOM_LLM_API_KEY or "no-key",
        model=settings.CUSTOM_LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        default_headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
    )
