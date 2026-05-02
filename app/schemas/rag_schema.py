"""
Pydantic schemas for the RAG ingestion endpoints.
"""

from pydantic import BaseModel
from typing import Optional


class EmbeddingConfig(BaseModel):
    """Embedding provider configuration forwarded from .NET backend SystemSettings."""
    provider: Optional[str] = None       # e.g. "cloudflare", "openai", "huggingface", "ollama"
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class QdrantConfig(BaseModel):
    """Qdrant connection configuration forwarded from .NET backend SystemSettings."""
    url: Optional[str] = None
    api_key: Optional[str] = None
    collection: Optional[str] = None


class IngestDocumentRequest(BaseModel):
    """Request body for POST /api/rag/ingest."""
    file_key: str               # DO Spaces key, e.g. "knowledge-base/{guid}/{filename}" — used for dedup + deletion
    file_name: str              # Original filename, used to detect format (.pdf / .docx / .md)
    file_bytes: str             # Base64-encoded file content
    content_type: str           # MIME type
    collection: str = "edunary_docs"
    embedding_config: Optional[EmbeddingConfig] = None
    qdrant_config: Optional[QdrantConfig] = None


class IngestDocumentResponse(BaseModel):
    success: bool
    chunk_count: int
    collection: str
    message: str


class DeleteDocumentRequest(BaseModel):
    """Request body for POST /api/rag/delete."""
    file_key: str               # The DO Spaces key used as metadata.file_key in Qdrant
    collection: str = "edunary_docs"
    qdrant_config: Optional[QdrantConfig] = None


class DeleteDocumentResponse(BaseModel):
    success: bool
    deleted_count: int
    message: str
