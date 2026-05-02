"""
Qdrant Dashboard service — admin management operations using qdrant-client SDK.

All public functions accept an optional qdrant_config that forwards connection
settings from the .NET backend SystemSettings (falls back to .env).
"""

from typing import Optional, Any

from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
)

from app.core.logging import get_logger
from app.embeddings.factory import build_qdrant_client
from app.schemas.qdrant_dashboard_schema import (
    CollectionSummary,
    CollectionListResponse,
    CollectionInfoResponse,
    VectorConfig,
    OptimizerStatus,
    PointRecord,
    PointListResponse,
    CountResponse,
    DeleteCollectionResponse,
)
from app.schemas.rag_schema import QdrantConfig

logger = get_logger(__name__)


def _build_filter(filter_key: Optional[str], filter_value: Optional[str]) -> Optional[Filter]:
    """Build a Qdrant Filter from simple key/value strings, or return None."""
    if filter_key and filter_value is not None:
        return Filter(
            must=[FieldCondition(key=filter_key, match=MatchValue(value=filter_value))]
        )
    return None


def _map_status(qdrant_status: Any) -> str:
    """Map Qdrant CollectionStatus enum to a simple traffic-light string."""
    s = str(qdrant_status).lower()
    if "green" in s or "ok" in s:
        return "green"
    if "yellow" in s or "optimizing" in s or "indexing" in s:
        return "yellow"
    return "red"


# ─── Collections ──────────────────────────────────────────────────────────────

def list_collections(qdrant_config: Optional[QdrantConfig] = None) -> CollectionListResponse:
    """List all Qdrant collections with summary stats."""
    client, _ = build_qdrant_client(qdrant_config)
    response = client.get_collections()

    summaries: list[CollectionSummary] = []
    for col in response.collections:
        try:
            info = client.get_collection(col.name)
            # Vector config — handle both named-dict and flat VectorParams
            vc = info.config.params.vectors
            if isinstance(vc, dict):
                # Named vectors: just take first
                first_key = next(iter(vc))
                vp = vc[first_key]
                vec_size = getattr(vp, "size", None)
                distance = str(getattr(vp, "distance", "")).split(".")[-1]
            else:
                vec_size = getattr(vc, "size", None)
                distance = str(getattr(vc, "distance", "")).split(".")[-1]

            summaries.append(CollectionSummary(
                name=col.name,
                status=_map_status(info.status),
                points_count=info.points_count or 0,
                vectors_count=getattr(info, "vectors_count", None),
                vector_size=vec_size,
                distance=distance,
                on_disk=getattr(vc, "on_disk", None),
                segments_count=info.segments_count,
            ))
        except Exception as e:
            logger.warning(f"Could not fetch info for collection '{col.name}': {e}")
            summaries.append(CollectionSummary(
                name=col.name,
                status="red",
                points_count=0,
            ))

    return CollectionListResponse(collections=summaries, total=len(summaries))


def get_collection_info(
    name: str,
    qdrant_config: Optional[QdrantConfig] = None,
) -> CollectionInfoResponse:
    """Get detailed info for a single collection."""
    client, _ = build_qdrant_client(qdrant_config)
    info = client.get_collection(name)

    # Parse vector config
    vc_raw = info.config.params.vectors
    if isinstance(vc_raw, dict):
        first_key = next(iter(vc_raw))
        vp = vc_raw[first_key]
        vec_cfg = VectorConfig(
            size=getattr(vp, "size", None),
            distance=str(getattr(vp, "distance", "")).split(".")[-1],
            on_disk=getattr(vp, "on_disk", None),
        )
    else:
        vec_cfg = VectorConfig(
            size=getattr(vc_raw, "size", None),
            distance=str(getattr(vc_raw, "distance", "")).split(".")[-1],
            on_disk=getattr(vc_raw, "on_disk", None),
        )

    # Optimizer status
    opt_raw = info.optimizer_status
    opt_status = OptimizerStatus(
        status=str(getattr(opt_raw, "ok", opt_raw)).lower() if opt_raw else None,
    )

    # Payload schema — map field types
    payload_schema: dict[str, Any] = {}
    if info.payload_schema:
        for field_name, field_info in info.payload_schema.items():
            payload_schema[field_name] = str(getattr(field_info, "data_type", field_info))
    # print(info)
    return CollectionInfoResponse(
        name=name,
        status=_map_status(info.status),
        points_count=info.points_count or 0,
        vectors_count=getattr(info, 'vectors_count', None),
        indexed_vectors_count=getattr(info, 'indexed_vectors_count', None),
        segments_count=info.segments_count,
        vector_config=vec_cfg,
        optimizer_status=opt_status,
        payload_schema=payload_schema,
    )


# ─── Points ───────────────────────────────────────────────────────────────────

def scroll_points(
    collection_name: str,
    limit: int = 20,
    offset: Optional[Any] = None,
    filter_key: Optional[str] = None,
    filter_value: Optional[str] = None,
    qdrant_config: Optional[QdrantConfig] = None,
) -> PointListResponse:
    """
    Scroll / paginate through points in a collection (payload only, no vectors).
    Returns next_offset=None when the last page is reached.
    """
    client, _ = build_qdrant_client(qdrant_config)
    scroll_filter = _build_filter(filter_key, filter_value)

    records, next_offset = client.scroll(
        collection_name=collection_name,
        scroll_filter=scroll_filter,
        limit=limit,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )

    points = [
        PointRecord(
            id=str(r.id),
            payload=r.payload or {},
        )
        for r in records
    ]

    # Serialize next_offset: Qdrant may return UUID str or int
    serialized_offset = str(next_offset) if next_offset is not None else None

    return PointListResponse(
        points=points,
        next_offset=serialized_offset,
    )


def count_points(
    collection_name: str,
    filter_key: Optional[str] = None,
    filter_value: Optional[str] = None,
    qdrant_config: Optional[QdrantConfig] = None,
) -> CountResponse:
    """Count points in a collection (with optional filter)."""
    client, _ = build_qdrant_client(qdrant_config)
    count_filter = _build_filter(filter_key, filter_value)

    result = client.count(
        collection_name=collection_name,
        count_filter=count_filter,
        exact=True,
    )
    return CountResponse(count=result.count)


def get_point(
    collection_name: str,
    point_id: str,
    qdrant_config: Optional[QdrantConfig] = None,
) -> Optional[PointRecord]:
    """Retrieve a single point by ID (UUID or int)."""
    client, _ = build_qdrant_client(qdrant_config)

    # Try UUID string first, then numeric
    pid: Any = point_id
    try:
        pid = int(point_id)
    except ValueError:
        pass

    records = client.retrieve(
        collection_name=collection_name,
        ids=[pid],
        with_payload=True,
        with_vectors=False,
    )
    if not records:
        return None
    r = records[0]
    return PointRecord(id=str(r.id), payload=r.payload or {})


# ─── Destructive Actions ──────────────────────────────────────────────────────

def delete_collection(
    name: str,
    qdrant_config: Optional[QdrantConfig] = None,
) -> DeleteCollectionResponse:
    """Delete an entire Qdrant collection."""
    client, _ = build_qdrant_client(qdrant_config)

    if not client.collection_exists(name):
        return DeleteCollectionResponse(
            success=False,
            message=f"Collection '{name}' does not exist.",
        )

    client.delete_collection(name)
    logger.info(f"Deleted Qdrant collection '{name}'")

    return DeleteCollectionResponse(
        success=True,
        message=f"Collection '{name}' deleted successfully.",
    )
