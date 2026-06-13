"""
Curriculum Embedding Controller — admin-facing endpoints for curriculum vector management.
All routes require HMAC authentication from the .NET backend.

Prefix: /api/curriculum
"""

import asyncio
from fastapi import APIRouter, Depends, HTTPException

from app.auth.dependencies import verify_request
from app.schemas.curriculum_embedding_schema import (
    CurriculumEmbedRequest,
    CurriculumEmbedResponse,
    DeleteCurriculumEmbeddingRequest,
    DeleteCurriculumEmbeddingResponse,
)
from app.services.curriculum_embedding_service import (
    embed_curriculum,
    delete_curriculum_embedding,
)
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(verify_request)])


@router.post("/embed", response_model=CurriculumEmbedResponse)
async def embed_curriculum_endpoint(request: CurriculumEmbedRequest):
    logger.info(
        f"Curriculum embed request: course_id={request.course_id}, "
        f"lectures_count={len(request.lectures)}"
    )

    try:
        total_chunks = await asyncio.to_thread(
            embed_curriculum,
            request.course_id,
            request.course_metadata,
            request.lectures,
            request.embedding_config,
            request.qdrant_config,
        )
    except Exception as e:
        logger.error(f"Failed to embed curriculum for course {request.course_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Curriculum embedding failed: {str(e)}")

    return CurriculumEmbedResponse(
        success=True,
        course_id=request.course_id,
        total_chunks=total_chunks,
        message=f"Successfully embedded curriculum for course ID {request.course_id}. Total {total_chunks} chunks stored.",
    )


@router.post("/delete", response_model=DeleteCurriculumEmbeddingResponse)
async def delete_curriculum_embedding_endpoint(request: DeleteCurriculumEmbeddingRequest):
    """
    Delete all curriculum points for a course ID from the edunary_curriculum Qdrant collection.
    Called by the .NET CourseCurriculumEmbeddingJobService background job.
    """
    logger.info(f"Curriculum delete embedding request: course_id={request.course_id}")

    try:
        await asyncio.to_thread(
            delete_curriculum_embedding,
            request.course_id,
            request.qdrant_config,
        )
    except Exception as e:
        logger.error(f"Failed to delete curriculum embedding for course {request.course_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

    return DeleteCurriculumEmbeddingResponse(
        success=True,
        course_id=request.course_id,
        message=f"Successfully deleted curriculum embedding for course ID {request.course_id}.",
    )
