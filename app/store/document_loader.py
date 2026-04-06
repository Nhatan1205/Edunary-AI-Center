"""
Document loader — load markdown documents, chunk them, and add to ChromaDB.
Refactored from services/document_loaders.py.

Usage (standalone script):
    python -m app.store.document_loader
"""

import os

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

from app.core.config import get_settings
from app.core.logging import get_logger
from app.embeddings.factory import create_embedding

logger = get_logger(__name__)


def generate_data_store() -> None:
    """Full pipeline: load documents → split → add to ChromaDB."""
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


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


def add_to_chroma(chunks: list[Document]) -> None:
    """Add document chunks to ChromaDB, skipping duplicates."""
    settings = get_settings()

    db = Chroma(
        collection_name=settings.CHROMA_COLLECTION,
        persist_directory=settings.CHROMA_PATH,
        embedding_function=create_embedding(),
    )

    # Generate IDs for each chunk
    chunks_with_ids = _calculate_chunk_ids(chunks)

    # Get existing IDs in DB
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])

    # Add only new documents
    new_chunks = []
    new_ids = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(chunk.metadata["id"])

    if new_chunks:
        db.add_documents(new_chunks, ids=new_ids)
        logger.info(f"Added {len(new_chunks)} new chunks to ChromaDB")
    else:
        logger.info("No new chunks to add")


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
