from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings loaded from .env file.
    All environment variables are centralized here.
    """

    # === App ===
    APP_TITLE: str = "Edunary AI Center"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # === LLM ===
    LLM_PROVIDER: str = "gemini"  # gemini | huggingface | ollama | custom
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 512

    # Gemini
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # HuggingFace
    HF_TOKEN: str = ""
    HUGGINGFACE_MODEL: str = "meta-llama/Llama-3.1-8B-Instruct"

    # Ollama
    OLLAMA_MODEL: str = "llama3.1"

    # Custom OpenAI-compatible API
    CUSTOM_LLM_BASE_URL: str = ""
    CUSTOM_LLM_API_KEY: str = ""
    CUSTOM_LLM_MODEL: str = ""

    # === Embedding ===
    EMBEDDING_PROVIDER: str = "huggingface"  # ollama | openai | huggingface | google
    HUGGINGFACE_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    OPENAI_API_KEY: str = ""
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    GOOGLE_EMBEDDING_MODEL: str = "models/embedding-001"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"

    # === Vector Store (Qdrant) ===
    QDRANT_URL: str = "http://localhost:6333"  # or Qdrant Cloud URL
    QDRANT_API_KEY: str = ""                    # leave empty for local Docker
    QDRANT_COLLECTION: str = "edunary_docs"
    DOCUMENTS_PATH: str = "document/edunary"

    # === MongoDB (Chat History) ===
    MONGO_URI: str = ""
    MONGO_DB_NAME: str = "chatbot"
    MONGO_COLLECTION_NAME: str = "chat_sessions"

    # === Edunary Backend API ===
    BACKEND_API_BASE_URL: str = "https://edunary.runasp.net"
    BACKEND_API_TIMEOUT: int = 30

    # === Auth ===
    JWT_SECRET_KEY: str = ""
    JWT_ALGORITHM: str = "HS256"
    AUTH_ENABLED: bool = False  # false = skip auth (dev mode)
    AI_CENTER_API_KEY: str = ""  # Static API key for .NET backend → AI Center calls

    # === Cloudflare Worker ===
    CF_BASE_URL: str = ""
    CF_API_KEY: str = ""
    CF_EMBEDDING_MODEL: str = "@cf/baai/bge-base-en-v1.5"  # 768-dim

    OPENAI_KEY: str = ""
    STT_MODEL: str = "whisper-1"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Return cached Settings singleton."""
    return Settings()
