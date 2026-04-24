"""Google Gemini LLM provider."""

from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import Settings


def create_gemini(settings: Settings):
    """Create a ChatGoogleGenerativeAI instance."""
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=settings.LLM_TEMPERATURE,
        convert_system_message_to_human=True,
    )
