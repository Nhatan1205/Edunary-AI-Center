"""
User Embedding Controller — admin-facing endpoints for user profile vector management.
All routes require HMAC authentication from the .NET backend.

Prefix: /api/user-embeddings

Sync service functions (embed/embed-batch/sync-status) are run via asyncio.to_thread()
so blocking HTTP calls to the embedding API never stall the FastAPI event loop.
"""

import asyncio
from fastapi import APIRouter, Depends, HTTPException

from app.auth.dependencies import verify_request
from app.schemas.user_embedding_schema import (
    EmbedUserProfileRequest,
    EmbedUserProfileResponse,
    BatchEmbedUsersRequest,
    BatchEmbedUsersResponse,
    SyncStatusRequest,
    SyncStatusResponse,
)
from app.services.user_embedding_service import (
    embed_user_profile,
    embed_users_batch,
    get_sync_status,
)
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(verify_request)])


@router.post("/embed", response_model=EmbedUserProfileResponse)
async def embed_user_profile_endpoint(request: EmbedUserProfileRequest):
    """
    Embed a single user profile into the edunary_users Qdrant collection.
    Called by the .NET UserEmbeddingJobService background job (ProcessUserProfileEmbeddingAsync).
    Idempotent — repeated calls with the same user_id will upsert (overwrite).
    """
    logger.info(
        f"User embed request: user_id={request.user_profile.user_id}, "
        f"name='{request.user_profile.full_name or request.user_profile.user_name}'"
    )

    try:
        await asyncio.to_thread(
            embed_user_profile,
            request.user_profile,
            request.embedding_config,
            request.qdrant_config,
        )
    except Exception as e:
        logger.error(f"Failed to embed user {request.user_profile.user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    return EmbedUserProfileResponse(
        success=True,
        user_id=request.user_profile.user_id,
        message=f"Successfully embedded user '{request.user_profile.full_name or request.user_profile.user_name}' (ID: {request.user_profile.user_id}).",
    )


@router.post("/embed-batch", response_model=BatchEmbedUsersResponse)
async def embed_users_batch_endpoint(request: BatchEmbedUsersRequest):
    """
    Batch embed multiple user profiles into the edunary_users Qdrant collection.
    Called by the .NET ProcessBatchUserProfileEmbeddingAsync background job.
    Processes in sub-batches of 20 for efficiency.
    """
    logger.info(f"Batch user embed request: {len(request.user_profiles)} profiles.")

    if not request.user_profiles:
        return BatchEmbedUsersResponse(
            success=True,
            total_embedded=0,
            failed_ids=[],
            message="No user profiles provided.",
        )

    try:
        total_embedded, failed_ids = await asyncio.to_thread(
            embed_users_batch,
            request.user_profiles,
            request.embedding_config,
            request.qdrant_config,
        )
    except Exception as e:
        logger.error(f"Batch user embedding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch embedding failed: {str(e)}")

    success = len(failed_ids) == 0
    message = (
        f"Successfully embedded {total_embedded}/{len(request.user_profiles)} user profiles."
        if success
        else f"Embedded {total_embedded}/{len(request.user_profiles)} user profiles. "
             f"Failed IDs: {failed_ids}."
    )

    return BatchEmbedUsersResponse(
        success=success,
        total_embedded=total_embedded,
        failed_ids=failed_ids,
        message=message,
    )


@router.post("/sync-status", response_model=SyncStatusResponse)
async def sync_status_endpoint(request: SyncStatusRequest):
    """
    Compare provided user IDs vs what is actually in Qdrant.
    Returns embedded_ids, missing_ids for admin dashboard display.
    Called by the .NET UserEmbeddings admin endpoint.
    """
    logger.info(f"User sync status request: {len(request.user_ids)} user IDs to check.")

    try:
        status = await asyncio.to_thread(
            get_sync_status,
            request.user_ids,
            request.qdrant_config,
        )
    except Exception as e:
        logger.error(f"User sync status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sync status check failed: {str(e)}")

    return SyncStatusResponse(**status)
