"""
Qdrant Dashboard controller — admin-facing collection + point management.
All routes are HMAC-protected (called by the .NET backend intermediary).
"""

from fastapi import APIRouter, Depends, HTTPException, Path

from app.auth.dependencies import verify_request
from app.core.logging import get_logger
from app.schemas.qdrant_dashboard_schema import (
    QdrantDashboardRequest,
    ScrollPointsRequest,
    CountPointsRequest,
    DeleteCollectionRequest,
    GetPointRequest,
    CollectionListResponse,
    CollectionInfoResponse,
    PointListResponse,
    CountResponse,
    DeleteCollectionResponse,
)
from app.services import qdrant_dashboard_service as svc

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(verify_request)])


# ─── Collections ──────────────────────────────────────────────────────────────

@router.post("/collections", response_model=CollectionListResponse)
async def list_collections(request: QdrantDashboardRequest):
    """
    List all Qdrant collections with summary stats.
    Called by .NET GET /api/QdrantDashboard/collections.
    """
    logger.info("Qdrant dashboard: listing collections")
    try:
        return svc.list_collections(request.qdrant_config)
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collections/{name}/info", response_model=CollectionInfoResponse)
async def get_collection_info(
    name: str = Path(..., description="Collection name"),
    request: QdrantDashboardRequest = None,
):
    """
    Get detailed info for a single collection.
    Called by .NET GET /api/QdrantDashboard/collections/{name}.
    """
    if request is None:
        request = QdrantDashboardRequest()
    logger.info(f"Qdrant dashboard: collection info for '{name}'")
    try:
        return svc.get_collection_info(name, request.qdrant_config)
    except Exception as e:
        logger.error(f"Failed to get collection info for '{name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Points ───────────────────────────────────────────────────────────────────

@router.post("/collections/{name}/points", response_model=PointListResponse)
async def scroll_points(
    name: str = Path(..., description="Collection name"),
    request: ScrollPointsRequest = None,
):
    """
    Scroll / paginate through points in a collection.
    Called by .NET GET /api/QdrantDashboard/collections/{name}/points.
    """
    if request is None:
        request = ScrollPointsRequest()
    logger.info(
        f"Qdrant dashboard: scroll points in '{name}' "
        f"(limit={request.limit}, offset={request.offset})"
    )
    try:
        return svc.scroll_points(
            collection_name=name,
            limit=request.limit,
            offset=request.offset,
            filter_key=request.filter_key,
            filter_value=request.filter_value,
            qdrant_config=request.qdrant_config,
        )
    except Exception as e:
        logger.error(f"Failed to scroll points in '{name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collections/{name}/count", response_model=CountResponse)
async def count_points(
    name: str = Path(..., description="Collection name"),
    request: CountPointsRequest = None,
):
    """
    Count points in a collection (with optional filter).
    """
    if request is None:
        request = CountPointsRequest()
    try:
        return svc.count_points(
            collection_name=name,
            filter_key=request.filter_key,
            filter_value=request.filter_value,
            qdrant_config=request.qdrant_config,
        )
    except Exception as e:
        logger.error(f"Failed to count points in '{name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collections/{name}/points/{point_id}")
async def get_point(
    name: str = Path(..., description="Collection name"),
    point_id: str = Path(..., description="Point UUID or integer ID"),
    request: GetPointRequest = None,
):
    """Retrieve a single point by ID."""
    if request is None:
        request = GetPointRequest()
    try:
        point = svc.get_point(name, point_id, request.qdrant_config)
        if point is None:
            raise HTTPException(status_code=404, detail=f"Point '{point_id}' not found in '{name}'")
        return {"success": True, "point": point}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get point '{point_id}' from '{name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Destructive Actions ──────────────────────────────────────────────────────

@router.post("/collections/{name}/delete", response_model=DeleteCollectionResponse)
async def delete_collection(
    name: str = Path(..., description="Collection name"),
    request: DeleteCollectionRequest = None,
):
    """
    Delete an entire Qdrant collection.
    Requires confirm=True in the request body.
    Called by .NET DELETE /api/QdrantDashboard/collections/{name}.
    """
    if request is None:
        request = DeleteCollectionRequest()
    if not request.confirm:
        raise HTTPException(status_code=400, detail="confirm must be true to delete a collection")
    logger.warning(f"Qdrant dashboard: DELETING collection '{name}'")
    try:
        return svc.delete_collection(name, request.qdrant_config)
    except Exception as e:
        logger.error(f"Failed to delete collection '{name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))
