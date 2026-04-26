"""
LLM Factory — creates an LLM instance based on Settings.LLM_PROVIDER.

Supported providers:
  - gemini       : Google Gemini (ChatGoogleGenerativeAI)
  - huggingface  : HuggingFace Router via OpenAI-compatible API
  - ollama       : Local Ollama (OllamaLLM)
  - custom       : Any OpenAI-compatible API endpoint

For requests coming from the .NET backend (which carry a model_config payload),
use `create_llm_from_config()` instead — it bypasses the factory switch and
directly creates a ChatOpenAI instance from the provided config.
"""

from langchain_openai import ChatOpenAI
from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_llm_instance = None


def create_llm(settings: Settings = None):
    """
    Create and cache a singleton LLM instance (reads from .env).
    Used by the chatbot service which runs independently from the .NET backend.
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


def create_llm_from_config(model_config: dict):
    """
    Create a one-off ChatOpenAI instance from a model_config dict sent by the .NET backend.

    Unlike `create_llm()` (which reads from .env and caches a singleton),
    this function creates a fresh instance per-request using the config
    values from the Edunary SystemSettings table.

    Every modern LLM provider exposes an OpenAI-compatible API
    (e.g. OpenAI, Azure, vLLM, LiteLLM, Ollama, HuggingFace Router),
    so we can always use ChatOpenAI here regardless of the actual provider.

    Args:
        model_config: dict with keys matching app.schemas.model_config.ModelConfig:
            - model_name : str
            - api_key    : str
            - api_base   : str  (OpenAI-compatible endpoint URL)
            - temperature: float (default 0.7)
            - max_tokens : int   (default 512)

    Returns:
        A LangChain ChatOpenAI instance ready to invoke.

    Raises:
        ValueError: If api_base is not provided.
    """
    api_base = model_config.get("api_base", "")
    if not api_base:
        raise ValueError(
            "model_config.api_base is required when using create_llm_from_config(). "
            "Ensure LLM_BaseUrl is configured in Edunary SystemSettings."
        )

    model_name = model_config.get("model_name", "")
    logger.info(f"Creating per-request LLM: model={model_name}, base={api_base}")

    return ChatOpenAI(
        base_url=api_base,
        api_key=model_config.get("api_key") or "no-key",
        model=model_name,
        temperature=model_config.get("temperature", 0.7),
        max_tokens=model_config.get("max_tokens", 512),
    )
