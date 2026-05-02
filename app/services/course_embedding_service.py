"""
Course Embedding Service — manages the `edunary_courses` Qdrant collection.

Handles:
  - Building searchable text from course fields
  - Embedding single / batch courses via upsert (idempotent UUID5 IDs)
  - Deleting a course point
  - Sync-status comparison (which course IDs are embedded vs missing)
  - Semantic similarity search for the chatbot

Collection name: edunary_courses (default, overridable via QdrantConfig)
Point ID strategy: UUID5(NAMESPACE_DNS, "edunary_course:<course_id>") — deterministic

Performance notes:
  - embed_course / embed_courses_batch generate vectors ONCE and upsert directly
    via QdrantClient.upsert(), avoiding a second embedding API round-trip.
  - Sync operations are run via asyncio.to_thread() so blocking HTTP calls
    (Cloudflare Workers AI) never block the FastAPI event loop.
"""

import json
import uuid
from typing import Optional

from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
)

from app.core.logging import get_logger
from app.embeddings.factory import build_embedding, build_qdrant_client
from app.schemas.course_embedding_schema import (
    CourseData,
    CourseSearchResult,
)
from app.schemas.rag_schema import EmbeddingConfig, QdrantConfig

logger = get_logger(__name__)

COURSE_COLLECTION = "edunary_courses"
_UUID_NAMESPACE = uuid.NAMESPACE_DNS


# ── Internal Helpers ─────────────────────────────────────────────────────────

def _course_point_id(course_id: int) -> str:
    """Derive a deterministic UUID5 string from a course integer ID."""
    return str(uuid.uuid5(_UUID_NAMESPACE, f"edunary_course:{course_id}"))


def _build_course_text(course: CourseData) -> str:
    """
    Compose a single searchable text string from course fields.
    This text is what gets embedded — richer text = better semantic search.
    """
    parts = []

    if course.title:
        parts.append(f"Title: {course.title}")
    if course.subtitle:
        parts.append(f"Subtitle: {course.subtitle}")
    if course.description:
        parts.append(f"Description: {course.description}")
    if course.learning_objectives:
        try:
            objectives = json.loads(course.learning_objectives)
            parts.append("Learning Objectives: " + "; ".join(objectives))
        except Exception:
            parts.append(f"Learning Objectives: {course.learning_objectives}")
    if course.requirements:
        try:
            reqs = json.loads(course.requirements)
            parts.append("Requirements: " + "; ".join(reqs))
        except Exception:
            parts.append(f"Requirements: {course.requirements}")
    if course.target_audience:
        try:
            audience = json.loads(course.target_audience)
            parts.append("Target Audience: " + "; ".join(audience))
        except Exception:
            parts.append(f"Target Audience: {course.target_audience}")
    if course.category_name:
        parts.append(f"Category: {course.category_name}")
    if course.topics:
        parts.append("Topics: " + ", ".join(course.topics))
    if course.level:
        parts.append(f"Level: {course.level}")
    if course.instructor_name:
        parts.append(f"Instructor: {course.instructor_name}")

    return "\n".join(parts)


def _course_to_payload(course: CourseData) -> dict:
    """
    Build the Qdrant point payload from course data.
    Stores all metadata so results can be returned without a DB roundtrip.
    """
    return {
        "course_id": course.course_id,
        "title": course.title,
        "subtitle": course.subtitle,
        "description": course.description,
        "category_name": course.category_name,
        "topics": course.topics or [],
        "level": course.level,
        "price": course.price,
        "image_url": course.image_url,
        "instructor_name": course.instructor_name,
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

def embed_course(
    course: CourseData,
    embedding_config: Optional[EmbeddingConfig] = None,
    qdrant_config: Optional[QdrantConfig] = None,
) -> None:
    """
    Embed a single course and upsert it into Qdrant.

    Generates the embedding vector ONCE and upserts directly via
    QdrantClient.upsert() — no second embedding API call.
    Idempotent: repeated calls with the same course_id overwrite the point.
    """
    embedding = build_embedding(embedding_config)
    client, _ = build_qdrant_client(qdrant_config)
    collection = (
        (qdrant_config.collection if qdrant_config else None) or COURSE_COLLECTION
    )

    text = _build_course_text(course)

    # Single embedding API call
    vector = embedding.embed_query(text)

    # Auto-create collection using the known dimension (no extra API call)
    _ensure_collection_with_dim(client, collection, len(vector))

    point_id = _course_point_id(course.course_id)
    payload = _course_to_payload(course)

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
        f"Embedded course {course.course_id} — '{course.title}' → point {point_id}"
    )


def embed_courses_batch(
    courses: list[CourseData],
    embedding_config: Optional[EmbeddingConfig] = None,
    qdrant_config: Optional[QdrantConfig] = None,
) -> tuple[int, list[int]]:
    """
    Batch embed multiple courses using a single embedding API call per sub-batch.
    Returns (total_embedded, failed_ids).
    Sub-batches of 20 to stay within token/request limits.
    """
    if not courses:
        return 0, []

    embedding = build_embedding(embedding_config)
    client, _ = build_qdrant_client(qdrant_config)
    collection = (
        (qdrant_config.collection if qdrant_config else None) or COURSE_COLLECTION
    )

    BATCH_SIZE = 20
    total_embedded = 0
    failed_ids: list[int] = []
    collection_created = False

    for batch_start in range(0, len(courses), BATCH_SIZE):
        batch = courses[batch_start: batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(courses) + BATCH_SIZE - 1) // BATCH_SIZE

        try:
            texts = [_build_course_text(c) for c in batch]

            # Single embedding API call for the whole sub-batch
            vectors = embedding.embed_documents(texts)

            # Auto-create collection on first successful embedding
            if not collection_created:
                _ensure_collection_with_dim(client, collection, len(vectors[0]))
                collection_created = True

            points = [
                PointStruct(
                    id=_course_point_id(c.course_id),
                    vector=vec,
                    payload=_course_to_payload(c),
                )
                for c, vec in zip(batch, vectors)
            ]

            client.upsert(collection_name=collection, points=points)

            total_embedded += len(batch)
            logger.info(
                f"Batch {batch_num}/{total_batches}: embedded {len(batch)} courses."
            )

        except Exception as e:
            logger.error(f"Batch {batch_num}/{total_batches} failed: {e}")
            failed_ids.extend([c.course_id for c in batch])

    logger.info(
        f"Batch embedding complete: {total_embedded} succeeded, {len(failed_ids)} failed."
    )
    return total_embedded, failed_ids


def delete_course_embedding(
    course_id: int,
    qdrant_config: Optional[QdrantConfig] = None,
) -> None:
    """Delete a single course point from the Qdrant collection."""
    client, _ = build_qdrant_client(qdrant_config)
    collection = (
        (qdrant_config.collection if qdrant_config else None) or COURSE_COLLECTION
    )

    if not client.collection_exists(collection):
        logger.warning(f"Collection '{collection}' does not exist. Nothing to delete.")
        return

    point_id = _course_point_id(course_id)
    client.delete(collection_name=collection, points_selector=[point_id])
    logger.info(f"Deleted course embedding: course_id={course_id}, point={point_id}")


def get_sync_status(
    course_ids: list[int],
    qdrant_config: Optional[QdrantConfig] = None,
) -> dict:
    """
    Compare the provided list of course IDs against what is actually in Qdrant.
    Returns a dict with embedded_ids, missing_ids, total_embedded, total_missing.
    """
    client, _ = build_qdrant_client(qdrant_config)
    collection = (
        (qdrant_config.collection if qdrant_config else None) or COURSE_COLLECTION
    )

    if not client.collection_exists(collection):
        return {
            "embedded_ids": [],
            "missing_ids": list(course_ids),
            "total_embedded": 0,
            "total_missing": len(course_ids),
        }

    # Build expected point IDs for the given course_ids
    expected_map = {_course_point_id(cid): cid for cid in course_ids}
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
        cid for pid, cid in expected_map.items() if pid in found_point_ids
    ]
    missing_ids = [
        cid for pid, cid in expected_map.items() if pid not in found_point_ids
    ]

    return {
        "embedded_ids": embedded_ids,
        "missing_ids": missing_ids,
        "total_embedded": len(embedded_ids),
        "total_missing": len(missing_ids),
    }


def search_courses(
    query: str,
    limit: int = 5,
    embedding_config: Optional[EmbeddingConfig] = None,
    qdrant_config: Optional[QdrantConfig] = None,
) -> list[CourseSearchResult]:
    """
    Perform semantic similarity search against the edunary_courses collection.
    Generates the query vector once, then calls client.query_points() directly.
    Returns a list of CourseSearchResult sorted by relevance score (descending).
    """
    embedding = build_embedding(embedding_config)
    client, _ = build_qdrant_client(qdrant_config)
    collection = (
        (qdrant_config.collection if qdrant_config else None) or COURSE_COLLECTION
    )

    if not client.collection_exists(collection):
        logger.warning(f"Collection '{collection}' does not exist — no results.")
        return []

    # Single embedding API call
    query_vector = embedding.embed_query(query)

    response = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )
    hits = response.points

    results: list[CourseSearchResult] = []
    for hit in hits:
        payload = hit.payload or {}
        course_id = payload.get("course_id")
        if course_id is None:
            continue
        results.append(
            CourseSearchResult(
                course_id=int(course_id),
                title=payload.get("title", ""),
                subtitle=payload.get("subtitle"),
                image_url=payload.get("image_url"),
                instructor_name=payload.get("instructor_name"),
                price=payload.get("price"),
                level=payload.get("level"),
                category_name=payload.get("category_name"),
                score=float(hit.score),
            )
        )

    logger.info(f"Course search '{query}': {len(results)} results found.")
    return results
