"""
Embedding factory — creates an embedding function based on Settings.EMBEDDING_PROVIDER.

Supported providers: ollama, openai, huggingface, google, cloudflare

Also exposes:
  - build_embedding(embedding_config)  — dynamic embedding from request config (falls back to .env)
  - build_qdrant_client(qdrant_config) — dynamic QdrantClient from request config (falls back to .env)
  - load_document(path, name)          — load PDF / DOCX / MD into LangChain Documents
  - chunk_documents(docs)              — split Documents with RecursiveCharacterTextSplitter
  - assign_chunk_ids(chunks, file_key) — stamp deterministic UUIDs + file_key metadata
"""

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def create_embedding(settings: Settings = None):
    """
    Get embedding function based on environment configuration.

    Set EMBEDDING_PROVIDER in .env file:
    - 'ollama'      : Local Ollama embeddings
    - 'openai'      : OpenAI embeddings
    - 'huggingface' : HuggingFace Inference API (remote)
    - 'google'      : Google Generative AI embeddings
    - 'cloudflare'  : Cloudflare Workers AI (OpenAI-compatible, uses CF_WORKER_URL)
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

    elif provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        return GoogleGenerativeAIEmbeddings(
            model=settings.GOOGLE_EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
        )

    elif provider == "cloudflare":
        from langchain_openai import OpenAIEmbeddings

        if not settings.CF_BASE_URL:
            raise ValueError("CF_BASE_URL not found in environment variables")
        if not settings.CF_API_KEY:
            raise ValueError("CF_API_KEY not found in environment variables")

        # Cloudflare Workers AI exposes an OpenAI-compatible /embeddings endpoint.
        # The base_url must end with /v1 (or your Worker routes it accordingly).
        return OpenAIEmbeddings(
            base_url=settings.CF_BASE_URL,
            api_key=settings.CF_API_KEY,
            model=settings.CF_EMBEDDING_MODEL,
            check_embedding_ctx_length=False
        )

    else:  # Default to Ollama
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=settings.OLLAMA_EMBEDDING_MODEL)


# ─── Dynamic helpers (config forwarded from .NET backend) ──────────────────────

def build_embedding(embedding_config=None):
    """
    Create an embedding function from a request-level config object.
    Falls back to .env Settings when the config field is missing or empty.

    embedding_config is expected to have optional attributes:
        provider, model_name, api_key, base_url
    """
    if embedding_config and getattr(embedding_config, "provider", None):
        settings = get_settings()

        class _DynamicSettings:
            EMBEDDING_PROVIDER      = embedding_config.provider
            OPENAI_API_KEY          = embedding_config.api_key      or settings.OPENAI_API_KEY
            OPENAI_EMBEDDING_MODEL  = embedding_config.model_name   or settings.OPENAI_EMBEDDING_MODEL
            HF_TOKEN                = embedding_config.api_key      or settings.HF_TOKEN
            HUGGINGFACE_EMBEDDING_MODEL = embedding_config.model_name or settings.HUGGINGFACE_EMBEDDING_MODEL
            GOOGLE_API_KEY          = embedding_config.api_key      or settings.GOOGLE_API_KEY
            GOOGLE_EMBEDDING_MODEL  = embedding_config.model_name   or settings.GOOGLE_EMBEDDING_MODEL
            OLLAMA_EMBEDDING_MODEL  = embedding_config.model_name   or settings.OLLAMA_EMBEDDING_MODEL
            CF_BASE_URL             = embedding_config.base_url     or settings.CF_BASE_URL
            CF_API_KEY              = embedding_config.api_key      or settings.CF_API_KEY
            CF_EMBEDDING_MODEL      = embedding_config.model_name   or settings.CF_EMBEDDING_MODEL

        return create_embedding(_DynamicSettings())

    return create_embedding()


def build_qdrant_client(qdrant_config=None):
    """
    Build a (QdrantClient, collection_name) tuple from a request-level config object.
    Falls back to .env Settings when config fields are missing.

    qdrant_config is expected to have optional attributes: url, api_key, collection
    """
    from qdrant_client import QdrantClient

    settings = get_settings()
    url        = (getattr(qdrant_config, "url",        None) if qdrant_config else None) or settings.QDRANT_URL
    api_key    = (getattr(qdrant_config, "api_key",    None) if qdrant_config else None) or settings.QDRANT_API_KEY or None
    collection = (getattr(qdrant_config, "collection", None) if qdrant_config else None) or settings.QDRANT_COLLECTION

    client = QdrantClient(url=url, api_key=api_key)
    return client, collection


def load_document(file_path: str, file_name: str):
    """
    Load a PDF, DOCX, or MD file from disk into a list of LangChain Document objects.
    """
    import os
    from langchain_community.document_loaders import (
        PyPDFLoader,
        Docx2txtLoader,
        UnstructuredMarkdownLoader,
    )

    ext = os.path.splitext(file_name)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".md":
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Only .pdf, .docx, .md are supported.")
    return loader.load()


_CHUNK_SIZE    = 800
_CHUNK_OVERLAP = 150


def chunk_documents(documents):
    """
    Split a list of Documents into chunks using RecursiveCharacterTextSplitter.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def assign_chunk_ids(chunks, file_key: str):
    """
    Stamp each chunk with:
      - metadata["id"]       — "<file_key>:<index>" string used for UUID5 derivation
      - metadata["file_key"] — stored in Qdrant payload to enable delete-by-file
    Returns the mutated chunk list.
    """
    for i, chunk in enumerate(chunks):
        chunk.metadata["id"]       = f"{file_key}:{i}"
        chunk.metadata["file_key"] = file_key
    return chunks
