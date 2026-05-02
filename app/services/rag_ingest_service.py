"""
RAG ingest service — receives document bytes, chunks, embeds, and upserts into Qdrant.
Also provides deletion of all Qdrant points for a given file_key.

This service is called by the rag_controller endpoints which are invoked
from the .NET backend background job (KnowledgeBaseJobService) via HMAC-authenticated POST.
"""

import base64
import os
import tempfile
import uuid

from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

from app.core.logging import get_logger
from app.embeddings.factory import (
    build_embedding,
    build_qdrant_client,
    load_document,
    chunk_documents,
    assign_chunk_ids,
)
from app.schemas.rag_schema import EmbeddingConfig, QdrantConfig

logger = get_logger(__name__)


def ingest_document(
    file_key: str,
    file_name: str,
    file_bytes_b64: str,
    content_type: str,
    collection: str,
    embedding_config: EmbeddingConfig = None,
    qdrant_config: QdrantConfig = None,
) -> int:
    """
    Full ingest pipeline:
      1. Decode base64 bytes → temp file
      2. Load document (pdf / docx / md)
      3. Chunk
      4. Embed + upsert into Qdrant (deterministic UUIDs for dedup)
    Returns the number of chunks upserted.
    """
    file_bytes = base64.b64decode(file_bytes_b64)
    ext = os.path.splitext(file_name)[1].lower()

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        logger.info(f"Loading document: {file_name} ({len(file_bytes)} bytes)")
        documents = load_document(tmp_path, file_name)
        chunks    = chunk_documents(documents)
        chunks    = assign_chunk_ids(chunks, file_key)

        logger.info(f"Chunked '{file_name}' into {len(chunks)} chunks. Embedding + upserting...")

        embedding         = build_embedding(embedding_config)
        client, coll_name = build_qdrant_client(qdrant_config)

        # Ensure collection exists
        if not client.collection_exists(coll_name):
            sample_vector = embedding.embed_query("init")
            client.create_collection(
                collection_name=coll_name,
                vectors_config=VectorParams(size=len(sample_vector), distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection '{coll_name}'")

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=coll_name,
            embedding=embedding,
        )

        # Deterministic UUID5 IDs for upsert deduplication
        qdrant_ids = [
            str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.metadata["id"]))
            for chunk in chunks
        ]
        vector_store.add_documents(chunks, ids=qdrant_ids)

        logger.info(f"Successfully upserted {len(chunks)} chunks for '{file_name}' into '{coll_name}'.")
        return len(chunks)

    finally:
        os.unlink(tmp_path)


def delete_document(
    file_key: str,
    collection: str,
    qdrant_config: QdrantConfig = None,
) -> int:
    """
    Delete all Qdrant points whose payload.file_key == file_key.
    Returns 0 (Qdrant delete API does not return an exact count).
    """
    client, coll_name = build_qdrant_client(qdrant_config)

    if not client.collection_exists(coll_name):
        logger.warning(f"Collection '{coll_name}' does not exist. Nothing to delete.")
        return 0

    filter_condition = Filter(
        must=[
            FieldCondition(
                key="metadata.file_key",
                match=MatchValue(value=file_key),
            )
        ]
    )

    client.delete(
        collection_name=coll_name,
        points_selector=filter_condition,
    )

    logger.info(f"Deleted Qdrant points for file_key='{file_key}' from collection '{coll_name}'.")
    return 0
