"""
LLM Factory — creates an LLM instance based on Settings.LLM_PROVIDER.

Supported providers:
  - gemini       : Google Gemini (ChatGoogleGenerativeAI)
  - huggingface  : HuggingFace Router via OpenAI-compatible API
  - ollama       : Local Ollama (OllamaLLM)
  - custom       : Any OpenAI-compatible API endpoint
"""

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_llm_instance = None


def create_llm(settings: Settings = None):
    """
    Create and cache a singleton LLM instance.
    Call this from services to get the LLM.
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    if settings is None:
        settings = get_settings()

    provider = settings.LLM_PROVIDER.lower()
    logger.info(f"Initializing LLM provider: {provider}")

    if provider == "gemini":
        from app.llm.providers.gemini_provider import create_gemini
        _llm_instance = create_gemini(settings)

    elif provider == "huggingface":
        from app.llm.providers.huggingface_provider import create_huggingface
        _llm_instance = create_huggingface(settings)

    elif provider == "ollama":
        from app.llm.providers.ollama_provider import create_ollama
        _llm_instance = create_ollama(settings)

    elif provider == "custom":
        from app.llm.providers.custom_api_provider import create_custom
        _llm_instance = create_custom(settings)

    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'. "
                         f"Supported: gemini, huggingface, ollama, custom")

    logger.info(f"LLM provider '{provider}' initialized successfully")
    return _llm_instance
