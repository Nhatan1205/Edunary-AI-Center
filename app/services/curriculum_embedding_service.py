"""
Curriculum Embedding Service — manages the `edunary_curriculum` Qdrant collection.

Handles:
  - Splitting lecture content into chunks (lecture-based chunking with word count limit)
  - Embedding curriculum lectures via upsert (idempotent UUID5 IDs)
  - Deleting all curriculum points for a course ID
  - Performing course-filtered similarity search to find related lectures (cross-section context)
"""

import uuid
from typing import Optional, List

from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.core.logging import get_logger
from app.embeddings.factory import build_embedding, build_qdrant_client
from app.schemas.curriculum_embedding_schema import (
    CurriculumCourseMetadata,
    CurriculumLectureData,
    CurriculumSearchResult,
)
from app.schemas.rag_schema import EmbeddingConfig, QdrantConfig

logger = get_logger(__name__)

CURRICULUM_COLLECTION = "edunary_curriculum"
_UUID_NAMESPACE = uuid.NAMESPACE_DNS


def _curriculum_point_id(course_id: int, item_id: str, sub_index: int) -> str:
    """Derive a deterministic UUID5 string for a curriculum lecture chunk."""
    return str(uuid.uuid5(_UUID_NAMESPACE, f"curriculum:{course_id}:{item_id}:{sub_index}"))


def _split_text(text: str, max_words: int = 2000, overlap: int = 200) -> List[str]:
    """Split text into chunks of max_words with overlap words."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start += (max_words - overlap)
    return chunks


def _ensure_collection_with_dim(client, collection_name: str, dim: int) -> None:
    """Create the Qdrant collection if it doesn't exist."""
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

def embed_curriculum(
    course_id: int,
    course_metadata: CurriculumCourseMetadata,
    lectures: List[CurriculumLectureData],
    embedding_config: Optional[EmbeddingConfig] = None,
    qdrant_config: Optional[QdrantConfig] = None,
) -> int:
    """
    Chunks, embeds and upserts curriculum lectures into Qdrant.
    Only embeds lectures that have changed or are new (delta embedding).
    Deletes lectures that were removed from the curriculum.
    Returns total count of chunks currently stored in Qdrant for this course.
    """
    if not lectures:
        delete_curriculum_embedding(course_id, qdrant_config)
        return 0

    embedding = build_embedding(embedding_config)
    client, _ = build_qdrant_client(qdrant_config)
    collection = (
        (qdrant_config.collection if qdrant_config else None) or CURRICULUM_COLLECTION
    )

    # 1. Fetch existing points for this course from Qdrant to detect changes
    existing_items = {}
    if client.collection_exists(collection):
        limit = 5000
        scroll_filter = Filter(
            must=[
                FieldCondition(
                    key="course_id",
                    match=MatchValue(value=course_id),
                )
            ]
        )
        points, _ = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            item_id = payload.get("item_id")
            sub_idx = payload.get("chunk_sub_index", 0)
            text = payload.get("content_text", "")
            if item_id:
                if item_id not in existing_items:
                    existing_items[item_id] = []
                existing_items[item_id].append((p.id, sub_idx, text))

        # Sort chunks by sub_idx to compare in order
        for item_id in existing_items:
            existing_items[item_id].sort(key=lambda x: x[1])

    # 2. Compare incoming lectures with existing points
    incoming_item_ids = {l.item_id for l in lectures}
    points_to_delete = []
    chunks_to_embed = []
    total_stored_chunks = 0

    added_items = []
    modified_items = []
    deleted_items = []
    unchanged_items = []

    # Delete lectures that are no longer in the curriculum
    for item_id, chunks in existing_items.items():
        if item_id not in incoming_item_ids:
            points_to_delete.extend([c[0] for c in chunks])
            deleted_items.append(item_id)

    for lecture in lectures:
        sub_chunks = _split_text(lecture.content_text, max_words=2000, overlap=200)
        total_sub_chunks = len(sub_chunks)
        total_stored_chunks += total_sub_chunks

        # Check if lecture has changed
        changed = True
        is_new = lecture.item_id not in existing_items
        
        if not is_new:
            old_chunks = existing_items[lecture.item_id]
            if len(old_chunks) == total_sub_chunks:
                # Compare content text of each chunk
                match = True
                for old_c, new_text in zip(old_chunks, sub_chunks):
                    if old_c[2] != new_text:
                        match = False
                        break
                if match:
                    changed = False

        if changed:
            if is_new:
                added_items.append(lecture.item_id)
            else:
                modified_items.append(lecture.item_id)
                # If changed, delete existing points for this item first (in case chunk count changed)
                points_to_delete.extend([c[0] for c in existing_items[lecture.item_id]])

            # Prepare new chunks for embedding
            for sub_idx, chunk_text in enumerate(sub_chunks):
                point_id = _curriculum_point_id(course_id, lecture.item_id, sub_idx)
                payload = {
                    "course_id": course_id,
                    "course_title": course_metadata.course_title,
                    "course_category": course_metadata.course_category or "",
                    "course_topics": course_metadata.course_topics or [],
                    "section_id": lecture.section_id,
                    "section_index": lecture.section_index,
                    "section_title": lecture.section_title,
                    "section_learning_objectives": lecture.section_learning_objectives or "",
                    "item_id": lecture.item_id,
                    "item_index": lecture.item_index,
                    "item_title": lecture.item_title,
                    "item_type": lecture.item_type,
                    "content_text": chunk_text,
                    "chunk_sub_index": sub_idx,
                    "total_sub_chunks": total_sub_chunks,
                    "word_count": len(chunk_text.split()),
                    "has_code_blocks": lecture.has_code_blocks,
                    "preceding_item_ids": lecture.preceding_item_ids,
                    "updated_at": course_metadata.updated_at or "",
                }
                chunks_to_embed.append((point_id, chunk_text, payload))
        else:
            unchanged_items.append(lecture.item_id)

    # 3. Perform deletions
    if points_to_delete and client.collection_exists(collection):
        client.delete(collection_name=collection, points_selector=points_to_delete)
        logger.info(f"Deleted {len(points_to_delete)} stale chunks for course {course_id}")

    # 4. Embed and Upsert new/changed chunks
    if chunks_to_embed:
        BATCH_SIZE = 20
        collection_created = False

        for i in range(0, len(chunks_to_embed), BATCH_SIZE):
            batch = chunks_to_embed[i : i + BATCH_SIZE]
            texts = [item[1] for item in batch]
            vectors = embedding.embed_documents(texts)

            if not collection_created:
                _ensure_collection_with_dim(client, collection, len(vectors[0]))
                collection_created = True

            points = [
                PointStruct(
                    id=item[0],
                    vector=vector,
                    payload=item[2],
                )
                for item, vector in zip(batch, vectors)
            ]
            client.upsert(collection_name=collection, points=points)

        logger.info(
            f"Curriculum sync complete for course {course_id}: "
            f"added={len(added_items)} {added_items}, "
            f"modified={len(modified_items)} {modified_items}, "
            f"deleted={len(deleted_items)} {deleted_items}, "
            f"unchanged={len(unchanged_items)} {unchanged_items}. "
            f"Embedded {len(chunks_to_embed)} new/changed chunks."
        )
    else:
        logger.info(
            f"No changes detected for course {course_id}: "
            f"added=0, modified=0, "
            f"deleted={len(deleted_items)} {deleted_items}, "
            f"unchanged={len(unchanged_items)} {unchanged_items}. "
            f"Skipped embedding."
        )

    # Save debug deltas to file
    # try:
    #     import os
    #     import json
    #     debug_dir = os.path.join(os.getcwd(), "debug_payloads")
    #     os.makedirs(debug_dir, exist_ok=True)
    #     debug_path = os.path.join(debug_dir, f"curriculum_delta_{course_id}.json")
    #     with open(debug_path, "w", encoding="utf-8") as f:
    #         json.dump({
    #             "course_id": course_id,
    #             "added_count": len(added_items),
    #             "added_items": added_items,
    #             "modified_count": len(modified_items),
    #             "modified_items": modified_items,
    #             "deleted_count": len(deleted_items),
    #             "deleted_items": deleted_items,
    #             "unchanged_count": len(unchanged_items),
    #             "unchanged_items": unchanged_items,
    #             "total_embedded_chunks": len(chunks_to_embed)
    #         }, f, indent=2)
    #     logger.info(f"Saved curriculum debug delta to: {debug_path}")
    # except Exception as ex:
    #     logger.warning(f"Failed to save debug delta to file: {ex}")

    return total_stored_chunks


def delete_curriculum_embedding(
    course_id: int,
    qdrant_config: Optional[QdrantConfig] = None,
) -> None:
    """Delete all curriculum lecture points for a specific course ID."""
    client, _ = build_qdrant_client(qdrant_config)
    collection = (
        (qdrant_config.collection if qdrant_config else None) or CURRICULUM_COLLECTION
    )

    if not client.collection_exists(collection):
        logger.warning(f"Collection '{collection}' does not exist. Nothing to delete.")
        return

    # Delete points using a filter
    client.delete(
        collection_name=collection,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="course_id",
                    match=MatchValue(value=course_id),
                )
            ]
        ),
    )
    logger.info(f"Deleted all curriculum embeddings for course_id={course_id}")


def search_related_lectures(
    query_text: str,
    course_id: int,
    exclude_item_ids: Optional[List[str]] = None,
    limit: int = 5,
    embedding_config: Optional[EmbeddingConfig] = None,
    qdrant_config: Optional[QdrantConfig] = None,
) -> List[CurriculumSearchResult]:
    """
    Search for related lectures in the curriculum of the specified course.
    Filters by course_id and optionally excludes certain item_ids (e.g. items in the current batch).
    """
    embedding = build_embedding(embedding_config)
    client, _ = build_qdrant_client(qdrant_config)
    collection = (
        (qdrant_config.collection if qdrant_config else None) or CURRICULUM_COLLECTION
    )

    if not client.collection_exists(collection):
        return []

    # Embedding the query
    query_vector = embedding.embed_query(query_text)

    # Filter by course ID
    must_conditions = [
        FieldCondition(
            key="course_id",
            match=MatchValue(value=course_id),
        )
    ]

    # Filter out excluded item IDs
    must_not_conditions = []
    if exclude_item_ids:
        for item_id in exclude_item_ids:
            must_not_conditions.append(
                FieldCondition(
                    key="item_id",
                    match=MatchValue(value=item_id),
                )
            )

    search_filter = Filter(
        must=must_conditions,
        must_not=must_not_conditions if must_not_conditions else None,
    )

    response = client.query_points(
        collection_name=collection,
        query=query_vector,
        query_filter=search_filter,
        limit=limit,
        with_payload=True,
    )

    results = []
    for hit in response.points:
        payload = hit.payload or {}
        results.append(
            CurriculumSearchResult(
                course_id=int(payload.get("course_id", 0)),
                section_id=payload.get("section_id", ""),
                section_title=payload.get("section_title", ""),
                item_id=payload.get("item_id", ""),
                item_title=payload.get("item_title", ""),
                item_type=payload.get("item_type", ""),
                content_text=payload.get("content_text", ""),
                score=float(hit.score),
            )
        )

    return results
