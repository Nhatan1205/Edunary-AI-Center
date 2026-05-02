"""
Course Embedding Controller — admin-facing endpoints for course vector management.
All routes require HMAC authentication from the .NET backend.

Prefix: /api/course-embeddings

Sync service functions (embed/delete/search) are run via asyncio.to_thread()
so blocking HTTP calls to the embedding API never stall the FastAPI event loop.
"""

import asyncio
from fastapi import APIRouter, Depends, HTTPException

from app.auth.dependencies import verify_request
from app.schemas.course_embedding_schema import (
    EmbedCourseRequest,
    EmbedCourseResponse,
    BatchEmbedCoursesRequest,
    BatchEmbedCoursesResponse,
    DeleteCourseEmbeddingRequest,
    DeleteCourseEmbeddingResponse,
    SyncStatusRequest,
    SyncStatusResponse,
    SearchCoursesRequest,
    SearchCoursesResponse,
)
from app.services.course_embedding_service import (
    embed_course,
    embed_courses_batch,
    delete_course_embedding,
    get_sync_status,
    search_courses,
)
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(verify_request)])


@router.post("/embed", response_model=EmbedCourseResponse)
async def embed_course_endpoint(request: EmbedCourseRequest):
    """
    Embed a single course into the edunary_courses Qdrant collection.
    Called by the .NET CourseEmbeddingJobService background job (ProcessCourseEmbeddingAsync).
    Idempotent — repeated calls with the same course_id will upsert (overwrite).
    """
    logger.info(
        f"Course embed request: course_id={request.course.course_id}, "
        f"title='{request.course.title}'"
    )

    try:
        await asyncio.to_thread(
            embed_course,
            request.course,
            request.embedding_config,
            request.qdrant_config,
        )
    except Exception as e:
        logger.error(f"Failed to embed course {request.course.course_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    return EmbedCourseResponse(
        success=True,
        course_id=request.course.course_id,
        message=f"Successfully embedded course '{request.course.title}' (ID: {request.course.course_id}).",
    )


@router.post("/embed-batch", response_model=BatchEmbedCoursesResponse)
async def embed_courses_batch_endpoint(request: BatchEmbedCoursesRequest):
    """
    Batch embed multiple courses into the edunary_courses Qdrant collection.
    Called by the .NET ProcessBatchCourseEmbeddingAsync background job.
    Processes in sub-batches of 20 for efficiency.
    """
    logger.info(f"Batch embed request: {len(request.courses)} courses.")

    if not request.courses:
        return BatchEmbedCoursesResponse(
            success=True,
            total_embedded=0,
            failed_ids=[],
            message="No courses provided.",
        )

    try:
        total_embedded, failed_ids = await asyncio.to_thread(
            embed_courses_batch,
            request.courses,
            request.embedding_config,
            request.qdrant_config,
        )
    except Exception as e:
        logger.error(f"Batch embedding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch embedding failed: {str(e)}")

    success = len(failed_ids) == 0
    message = (
        f"Successfully embedded {total_embedded}/{len(request.courses)} courses."
        if success
        else f"Embedded {total_embedded}/{len(request.courses)} courses. "
             f"Failed IDs: {failed_ids}."
    )

    return BatchEmbedCoursesResponse(
        success=success,
        total_embedded=total_embedded,
        failed_ids=failed_ids,
        message=message,
    )


@router.post("/delete", response_model=DeleteCourseEmbeddingResponse)
async def delete_course_embedding_endpoint(request: DeleteCourseEmbeddingRequest):
    """
    Delete a course embedding from the edunary_courses Qdrant collection.
    Called by the .NET ProcessCourseEmbeddingDeletionAsync background job.
    """
    logger.info(f"Course delete embedding request: course_id={request.course_id}")

    try:
        await asyncio.to_thread(
            delete_course_embedding,
            request.course_id,
            request.qdrant_config,
        )
    except Exception as e:
        logger.error(f"Failed to delete embedding for course {request.course_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

    return DeleteCourseEmbeddingResponse(
        success=True,
        course_id=request.course_id,
        message=f"Successfully deleted embedding for course ID {request.course_id}.",
    )


@router.post("/sync-status", response_model=SyncStatusResponse)
async def sync_status_endpoint(request: SyncStatusRequest):
    """
    Compare provided course IDs vs what is actually in Qdrant.
    Returns embedded_ids, missing_ids for admin dashboard display.
    Called by the .NET CourseEmbeddings admin endpoint.
    """
    logger.info(f"Sync status request: {len(request.course_ids)} course IDs to check.")

    try:
        status = await asyncio.to_thread(
            get_sync_status,
            request.course_ids,
            request.qdrant_config,
        )
    except Exception as e:
        logger.error(f"Sync status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sync status check failed: {str(e)}")

    return SyncStatusResponse(**status)


@router.post("/search", response_model=SearchCoursesResponse)
async def search_courses_endpoint(request: SearchCoursesRequest):
    """
    Perform semantic similarity search against the edunary_courses collection.
    Used internally by course_service.py chatbot handler.
    """
    logger.info(f"Course search request: query='{request.query}', limit={request.limit}")

    try:
        results = await asyncio.to_thread(
            search_courses,
            request.query,
            request.limit,
            request.embedding_config,
            request.qdrant_config,
        )
    except Exception as e:
        logger.error(f"Course search failed for query '{request.query}': {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    return SearchCoursesResponse(results=results, total=len(results))
