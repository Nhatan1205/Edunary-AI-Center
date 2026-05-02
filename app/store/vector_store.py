"""
Vector store — Qdrant operations for RAG similarity search.
Refactored from services/query_data.py (previously ChromaDB).

Supports both local Docker (QDRANT_API_KEY empty) and Qdrant Cloud (QDRANT_API_KEY set).
"""

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from app.core.config import get_settings
from app.core.logging import get_logger
from app.embeddings.factory import create_embedding

logger = get_logger(__name__)

_db_instance: QdrantVectorStore | None = None


def _get_db() -> QdrantVectorStore:
    """Get or create the Qdrant vector store instance (singleton)."""
    global _db_instance
    if _db_instance is None:
        settings = get_settings()
        embedding = create_embedding()

        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,  # None = no auth (local Docker)
        )

        # Auto-create collection if it doesn't exist yet.
        # Vector dimension is inferred from the configured embedding model.
        if not client.collection_exists(settings.QDRANT_COLLECTION):
            sample_vector = embedding.embed_query("init")
            client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=len(sample_vector),
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                f"Created Qdrant collection '{settings.QDRANT_COLLECTION}' "
                f"(dim={len(sample_vector)})"
            )

        _db_instance = QdrantVectorStore(
            client=client,
            collection_name=settings.QDRANT_COLLECTION,
            embedding=embedding,
        )
        logger.info(
            f"Qdrant loaded: url='{settings.QDRANT_URL}', "
            f"collection='{settings.QDRANT_COLLECTION}'"
        )
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
