"""
Vector store — ChromaDB operations for RAG similarity search.
Refactored from services/query_data.py.
"""

from langchain_chroma import Chroma

from app.core.config import get_settings
from app.core.logging import get_logger
from app.embeddings.factory import create_embedding

logger = get_logger(__name__)

_db_instance = None


def _get_db() -> Chroma:
    """Get or create the ChromaDB instance (singleton)."""
    global _db_instance
    if _db_instance is None:
        settings = get_settings()
        _db_instance = Chroma(
            collection_name=settings.CHROMA_COLLECTION,
            persist_directory=settings.CHROMA_PATH,
            embedding_function=create_embedding(),
        )
        logger.info(f"ChromaDB loaded: collection='{settings.CHROMA_COLLECTION}', path='{settings.CHROMA_PATH}'")
    return _db_instance


def similarity_search(query: str, k: int = 5) -> list:
    """
    Search the vector store for similar documents.

    Args:
        query: The search query text.
        k: Number of results to return.

    Returns:
        list of (Document, score) tuples.
    """
    db = _get_db()
    return db.similarity_search_with_score(query, k=k)
