"""
Document loader — load markdown documents, chunk them, and upsert into Qdrant.
Refactored from services/document_loaders.py (previously ChromaDB).

Usage (standalone script):
    python -m app.store.document_loader
"""

import os
import uuid

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger
from app.store.vector_store import _get_db

logger = get_logger(__name__)


def generate_data_store() -> None:
    """Full pipeline: load documents → split → upsert into Qdrant."""
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_qdrant(chunks)


def load_documents() -> list[Document]:
    """Load all markdown documents from the configured documents path."""
    settings = get_settings()
    data_path = settings.DOCUMENTS_PATH
    logger.info(f"Loading documents from: {data_path}")

    loader = DirectoryLoader(data_path, glob="*.md")
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} documents")
    return docs


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks


def add_to_qdrant(chunks: list[Document]) -> None:
    """
    Upsert document chunks into Qdrant.

    Qdrant requires point IDs to be either an unsigned integer or a UUID.
    We derive a deterministic UUID5 from the human-readable chunk string ID
    so that re-running produces the same UUID → upsert dedup works correctly.
    The original string ID is preserved in metadata['id'] for source tracking.
    """
    chunks_with_ids = _calculate_chunk_ids(chunks)

    # Convert string IDs → deterministic UUID5
    qdrant_ids = [
        str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.metadata["id"]))
        for chunk in chunks_with_ids
    ]

    db = _get_db()
    db.add_documents(chunks_with_ids, ids=qdrant_ids)
    logger.info(f"Upserted {len(chunks_with_ids)} chunks into Qdrant")


def _calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    """Calculate unique IDs for each chunk based on source and page."""
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks


if __name__ == "__main__":
    generate_data_store()
