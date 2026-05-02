"""
Pydantic schemas for the Qdrant Dashboard admin endpoints.
"""

from pydantic import BaseModel
from typing import Optional, Any

from app.schemas.rag_schema import QdrantConfig


# ─── Base Request ─────────────────────────────────────────────────────────────

class QdrantDashboardRequest(BaseModel):
    """Base request with optional dynamic Qdrant config."""
    qdrant_config: Optional[QdrantConfig] = None


class ScrollPointsRequest(QdrantDashboardRequest):
    """Request for paginated point browsing."""
    limit: int = 20
    offset: Optional[str] = None        # Qdrant next_page_offset (opaque, serialized as str)
    filter_key: Optional[str] = None    # e.g. "metadata.file_key"
    filter_value: Optional[str] = None  # e.g. "knowledge-base/..."


class CountPointsRequest(QdrantDashboardRequest):
    """Request for point count (with optional filter)."""
    filter_key: Optional[str] = None
    filter_value: Optional[str] = None


class DeleteCollectionRequest(QdrantDashboardRequest):
    """Request for collection deletion."""
    confirm: bool = False               # Must be True for deletion to proceed


class GetPointRequest(QdrantDashboardRequest):
    """Request to retrieve a single point by ID."""
    pass


# ─── Response Models ──────────────────────────────────────────────────────────

class CollectionSummary(BaseModel):
    """Summary of a single collection for the overview grid."""
    name: str
    status: str                         # "green" | "yellow" | "red" (mapped from Qdrant status)
    points_count: int
    vectors_count: Optional[int] = None
    vector_size: Optional[int] = None
    distance: Optional[str] = None
    on_disk: Optional[bool] = None
    segments_count: Optional[int] = None


class CollectionListResponse(BaseModel):
    success: bool = True
    collections: list[CollectionSummary]
    total: int


class VectorConfig(BaseModel):
    size: Optional[int] = None
    distance: Optional[str] = None
    on_disk: Optional[bool] = None


class OptimizerStatus(BaseModel):
    status: Optional[str] = None
    optimizations_as_of: Optional[str] = None


class CollectionInfoResponse(BaseModel):
    success: bool = True
    name: str
    status: str
    points_count: int
    vectors_count: Optional[int] = None
    indexed_vectors_count: Optional[int] = None
    segments_count: Optional[int] = None
    vector_config: Optional[VectorConfig] = None
    optimizer_status: Optional[OptimizerStatus] = None
    payload_schema: Optional[dict[str, Any]] = None


class PointRecord(BaseModel):
    id: Any                             # UUID str or int
    payload: Optional[dict[str, Any]] = None


class PointListResponse(BaseModel):
    success: bool = True
    points: list[PointRecord]
    next_offset: Optional[Any] = None   # None when last page reached
    total: Optional[int] = None         # filled when filter + count available


class CountResponse(BaseModel):
    success: bool = True
    count: int


class DeleteCollectionResponse(BaseModel):
    success: bool
    message: str
