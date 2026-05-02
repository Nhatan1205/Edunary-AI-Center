"""
User Embedding Service — manages the `edunary_users` Qdrant collection.

Handles:
  - Building searchable text from user profile fields
  - Embedding single / batch user profiles via upsert (idempotent UUID5 IDs)
  - Sync-status comparison (which user IDs are embedded vs missing)

Collection name: edunary_users (default, overridable via QdrantConfig)
Point ID strategy: UUID5(NAMESPACE_DNS, "edunary_user:<user_id>") — deterministic

Performance notes:
  - embed_user_profile / embed_users_batch generate vectors ONCE and upsert directly
    via QdrantClient.upsert(), avoiding a second embedding API round-trip.
  - Sync operations are run via asyncio.to_thread() so blocking HTTP calls
    (Cloudflare Workers AI) never block the FastAPI event loop.
"""

import uuid
from typing import Optional

from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from app.core.logging import get_logger
from app.embeddings.factory import build_embedding, build_qdrant_client
from app.schemas.user_embedding_schema import UserProfileData
from app.schemas.rag_schema import EmbeddingConfig, QdrantConfig

logger = get_logger(__name__)

USER_COLLECTION = "edunary_users"
_UUID_NAMESPACE = uuid.NAMESPACE_DNS


# ── Internal Helpers ─────────────────────────────────────────────────────────

def _user_point_id(user_id: str) -> str:
    """Derive a deterministic UUID5 string from a user GUID string."""
    return str(uuid.uuid5(_UUID_NAMESPACE, f"edunary_user:{user_id}"))


def _build_user_text(profile: UserProfileData) -> str:
    """
    Compose a single searchable text string from user profile fields.
    This text is what gets embedded — richer text = better semantic search.
    """
    parts = []

    if profile.user_name:
        parts.append(f"Username: {profile.user_name}")
    if profile.full_name:
        parts.append(f"Full Name: {profile.full_name}")
    if profile.goal:
        parts.append(f"Learning Goal: {profile.goal}")
    if profile.skill_level:
        parts.append(f"Skill Level: {profile.skill_level}")
    if profile.preferred_categories:
        parts.append("Preferred Categories: " + ", ".join(profile.preferred_categories))
    if profile.preferred_topics:
        parts.append("Preferred Topics: " + ", ".join(profile.preferred_topics))
    if profile.weekly_hours is not None:
        parts.append(f"Weekly Study Hours: {profile.weekly_hours}")

    return "\n".join(parts)


def _user_to_payload(profile: UserProfileData) -> dict:
    """
    Build the Qdrant point payload from user profile data.
    Stores all metadata so results can be returned without a DB roundtrip.
    """
    return {
        "user_id": profile.user_id,
        "user_name": profile.user_name,
        "full_name": profile.full_name,
        "goal": profile.goal,
        "skill_level": profile.skill_level,
        "preferred_categories": profile.preferred_categories or [],
        "preferred_topics": profile.preferred_topics or [],
        "weekly_hours": profile.weekly_hours,
    }


def _ensure_collection_with_dim(client, collection_name: str, dim: int) -> None:
    """
    Create the Qdrant collection if it doesn't exist, using a known vector dim.
    Does NOT call the embedding API — caller must supply the dim.
    """
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Created Qdrant collection '{collection_name}' (dim={dim})")


# ── Public API (sync — called via asyncio.to_thread from the controller) ──────

def embed_user_profile(
    profile: UserProfileData,
    embedding_config: Optional[EmbeddingConfig] = None,
    qdrant_config: Optional[QdrantConfig] = None,
) -> None:
    """
    Embed a single user profile and upsert it into Qdrant.

    Generates the embedding vector ONCE and upserts directly via
    QdrantClient.upsert() — no second embedding API call.
    Idempotent: repeated calls with the same user_id overwrite the point.
    """
    embedding = build_embedding(embedding_config)
    client, _ = build_qdrant_client(qdrant_config)
    collection = (
        (qdrant_config.collection if qdrant_config else None) or USER_COLLECTION
    )

    text = _build_user_text(profile)

    # Single embedding API call
    vector = embedding.embed_query(text)

    # Auto-create collection using the known dimension (no extra API call)
    _ensure_collection_with_dim(client, collection, len(vector))

    point_id = _user_point_id(profile.user_id)
    payload = _user_to_payload(profile)

    client.upsert(
        collection_name=collection,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        ],
    )

    logger.info(
        f"Embedded user {profile.user_id} — '{profile.full_name or profile.user_name}' → point {point_id}"
    )


def embed_users_batch(
    profiles: list[UserProfileData],
    embedding_config: Optional[EmbeddingConfig] = None,
    qdrant_config: Optional[QdrantConfig] = None,
) -> tuple[int, list[str]]:
    """
    Batch embed multiple user profiles using a single embedding API call per sub-batch.
    Returns (total_embedded, failed_ids).
    Sub-batches of 20 to stay within token/request limits.
    """
    if not profiles:
        return 0, []

    embedding = build_embedding(embedding_config)
    client, _ = build_qdrant_client(qdrant_config)
    collection = (
        (qdrant_config.collection if qdrant_config else None) or USER_COLLECTION
    )

    BATCH_SIZE = 20
    total_embedded = 0
    failed_ids: list[str] = []
    collection_created = False

    for batch_start in range(0, len(profiles), BATCH_SIZE):
        batch = profiles[batch_start: batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(profiles) + BATCH_SIZE - 1) // BATCH_SIZE

        try:
            texts = [_build_user_text(p) for p in batch]

            # Single embedding API call for the whole sub-batch
            vectors = embedding.embed_documents(texts)

            # Auto-create collection on first successful embedding
            if not collection_created:
                _ensure_collection_with_dim(client, collection, len(vectors[0]))
                collection_created = True

            points = [
                PointStruct(
                    id=_user_point_id(p.user_id),
                    vector=vec,
                    payload=_user_to_payload(p),
                )
                for p, vec in zip(batch, vectors)
            ]

            client.upsert(collection_name=collection, points=points)

            total_embedded += len(batch)
            logger.info(
                f"Batch {batch_num}/{total_batches}: embedded {len(batch)} user profiles."
            )

        except Exception as e:
            logger.error(f"Batch {batch_num}/{total_batches} failed: {e}")
            failed_ids.extend([p.user_id for p in batch])

    logger.info(
        f"Batch user embedding complete: {total_embedded} succeeded, {len(failed_ids)} failed."
    )
    return total_embedded, failed_ids


def get_sync_status(
    user_ids: list[str],
    qdrant_config: Optional[QdrantConfig] = None,
) -> dict:
    """
    Compare the provided list of user IDs against what is actually in Qdrant.
    Returns a dict with embedded_ids, missing_ids, total_embedded, total_missing.
    """
    client, _ = build_qdrant_client(qdrant_config)
    collection = (
        (qdrant_config.collection if qdrant_config else None) or USER_COLLECTION
    )

    if not client.collection_exists(collection):
        return {
            "embedded_ids": [],
            "missing_ids": list(user_ids),
            "total_embedded": 0,
            "total_missing": len(user_ids),
        }

    # Build expected point IDs for the given user_ids
    expected_map = {_user_point_id(uid): uid for uid in user_ids}
    expected_point_ids = list(expected_map.keys())

    # Retrieve those specific points (payload only, no vectors)
    retrieved = client.retrieve(
        collection_name=collection,
        ids=expected_point_ids,
        with_payload=False,
        with_vectors=False,
    )
    found_point_ids = {str(r.id) for r in retrieved}

    embedded_ids = [
        uid for pid, uid in expected_map.items() if pid in found_point_ids
    ]
    missing_ids = [
        uid for pid, uid in expected_map.items() if pid not in found_point_ids
    ]

    return {
        "embedded_ids": embedded_ids,
        "missing_ids": missing_ids,
        "total_embedded": len(embedded_ids),
        "total_missing": len(missing_ids),
    }
