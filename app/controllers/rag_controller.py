"""
RAG controller — admin-facing document ingestion and deletion endpoints.
All routes require HMAC authentication from the .NET backend.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.auth.dependencies import verify_request
from app.schemas.rag_schema import (
    IngestDocumentRequest,
    IngestDocumentResponse,
    DeleteDocumentRequest,
    DeleteDocumentResponse,
)
from app.services.rag_ingest_service import ingest_document, delete_document
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(verify_request)])


@router.post("/ingest", response_model=IngestDocumentResponse)
async def ingest_document_endpoint(request: IngestDocumentRequest):
    """
    Receive a document (base64-encoded), chunk it, embed it, and upsert into Qdrant.
    Called by the .NET KnowledgeBaseJobService Hangfire background job.
    """
    logger.info(
        f"RAG ingest request: file='{request.file_name}', "
        f"collection='{request.collection}', key='{request.file_key}'"
    )

    try:
        chunk_count = ingest_document(
            file_key=request.file_key,
            file_name=request.file_name,
            file_bytes_b64=request.file_bytes,
            content_type=request.content_type,
            collection=request.collection,
            embedding_config=request.embedding_config,
            qdrant_config=request.qdrant_config,
        )
    except ValueError as e:
        logger.error(f"Ingest validation error for '{request.file_name}': {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Ingest failed for '{request.file_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    return IngestDocumentResponse(
        success=True,
        chunk_count=chunk_count,
        collection=request.collection,
        message=f"Successfully ingested '{request.file_name}' into '{request.collection}' ({chunk_count} chunks).",
    )


@router.post("/delete", response_model=DeleteDocumentResponse)
async def delete_document_endpoint(request: DeleteDocumentRequest):
    """
    Delete all Qdrant points for the given file_key.
    Called by the .NET DeleteKnowledgeDocumentCommand before removing the file from DO Spaces.
    """
    logger.info(
        f"RAG delete request: file_key='{request.file_key}', collection='{request.collection}'"
    )

    try:
        deleted_count = delete_document(
            file_key=request.file_key,
            collection=request.collection,
            qdrant_config=request.qdrant_config,
        )
    except Exception as e:
        logger.error(f"Delete failed for file_key='{request.file_key}': {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

    return DeleteDocumentResponse(
        success=True,
        deleted_count=deleted_count,
        message=f"Successfully deleted chunks for '{request.file_key}' from '{request.collection}'.",
    )
