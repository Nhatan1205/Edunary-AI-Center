"""
Embedding factory — creates an embedding function based on Settings.EMBEDDING_PROVIDER.

Supported providers: ollama, openai, huggingface, google
"""

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def create_embedding(settings: Settings = None):
    """
    Get embedding function based on environment configuration.

    Set EMBEDDING_PROVIDER in .env file:
    - 'ollama' (default): Use local Ollama embeddings
    - 'openai': Use OpenAI embeddings
    - 'huggingface': Use Hugging Face Inference API (online, no download)
    - 'google': Use Google Generative AI embeddings
    """
    if settings is None:
        settings = get_settings()

    provider = settings.EMBEDDING_PROVIDER.lower()
    logger.info(f"Initializing embedding provider: {provider}")

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        return OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
        )

    elif provider == "huggingface":
        from langchain_huggingface import HuggingFaceEndpointEmbeddings

        if not settings.HF_TOKEN:
            raise ValueError("HF_TOKEN not found in environment variables")

        return HuggingFaceEndpointEmbeddings(
            model=settings.HUGGINGFACE_EMBEDDING_MODEL,
            huggingfacehub_api_token=settings.HF_TOKEN,
        )

    elif provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        return GoogleGenerativeAIEmbeddings(
            model=settings.GOOGLE_EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
        )

    else:  # Default to Ollama
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=settings.OLLAMA_EMBEDDING_MODEL)
