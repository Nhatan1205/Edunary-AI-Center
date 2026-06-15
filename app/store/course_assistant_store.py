"""
Course Assistant History Store — MongoDB persistence for two collections:
  - course_assistant_conversations: one doc per (user_id, course_id)
  - course_assistant_messages:      one doc per message

Cursor pagination uses MongoDB ObjectId ordering (time-sortable).
Completely independent from chat_history_store.py (chatbot).
"""

from datetime import datetime, timezone
from typing import Optional, List

from bson import ObjectId
from pymongo import MongoClient, DESCENDING, ASCENDING
from pymongo.collection import Collection

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_settings = get_settings()
_mongo_client: MongoClient = MongoClient(_settings.MONGO_URI)
_db = _mongo_client[_settings.MONGO_DB_NAME]

_CONV_COLLECTION = "course_assistant_conversations"
_MSG_COLLECTION = "course_assistant_messages"


def _conversations() -> Collection:
    return _db[_CONV_COLLECTION]


def _messages() -> Collection:
    return _db[_MSG_COLLECTION]


def _ensure_indexes() -> None:
    """Create indexes once on first use (idempotent)."""
    _conversations().create_index(
        [("user_id", ASCENDING), ("course_id", ASCENDING)],
        unique=True,
        name="uq_user_course",
    )
    _messages().create_index(
        [("conversation_id", ASCENDING), ("_id", DESCENDING)],
        name="idx_conv_id_desc",
    )


_ensure_indexes()


# ── Conversation CRUD ─────────────────────────────────────────────────────────

def get_or_create_conversation(user_id: str, course_id: int, course_title: str = "") -> str:
    """
    Return the existing conversation _id (as str) for this (user_id, course_id),
    or create a new one if it doesn't exist.
    """
    existing = _conversations().find_one({"user_id": user_id, "course_id": course_id})
    if existing:
        return str(existing["_id"])

    result = _conversations().insert_one({
        "user_id": user_id,
        "course_id": course_id,
        "course_title": course_title,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    })
    logger.info(f"Created conversation for user={user_id} course={course_id}")
    return str(result.inserted_id)


def _touch_conversation(conversation_id: str) -> None:
    """Update updated_at timestamp on conversation."""
    _conversations().update_one(
        {"_id": ObjectId(conversation_id)},
        {"$set": {"updated_at": datetime.now(timezone.utc)}},
    )


def clear_conversation(user_id: str, course_id: int) -> None:
    """Delete conversation document and ALL its messages."""
    conv = _conversations().find_one({"user_id": user_id, "course_id": course_id})
    if not conv:
        return
    conv_id = conv["_id"]
    _messages().delete_many({"conversation_id": conv_id})
    _conversations().delete_one({"_id": conv_id})
    logger.info(f"Cleared conversation for user={user_id} course={course_id}")


# ── Message CRUD ──────────────────────────────────────────────────────────────

def add_message(
    conversation_id: str,
    role: str,
    content: str,
    content_id: Optional[str] = None,
    content_type: Optional[str] = None,
    sources: Optional[List[str]] = None,
) -> str:
    """Insert a message document and bump conversation.updated_at."""
    doc = {
        "conversation_id": ObjectId(conversation_id),
        "role": role,
        "content": content,
        "content_id": content_id,
        "content_type": content_type,
        "sources": sources,
        "created_at": datetime.now(timezone.utc),
    }
    result = _messages().insert_one(doc)
    _touch_conversation(conversation_id)
    return str(result.inserted_id)


def get_recent_messages(conversation_id: str, limit: int = 10) -> List[dict]:
    """
    Fetch the most recent `limit` messages for LLM context.
    Returns oldest-first so the LLM sees them in chronological order.
    """
    docs = list(
        _messages()
        .find({"conversation_id": ObjectId(conversation_id)})
        .sort("_id", DESCENDING)
        .limit(limit)
    )
    docs.reverse()  # oldest first for LLM prompt
    return docs


def get_messages_paginated(
    conversation_id: str,
    cursor: Optional[str],
    page_size: int = 20,
) -> dict:
    """
    Cursor-paginated message fetch (newest-first).

    cursor: ObjectId string of the OLDEST message currently visible in the UI.
            Pass None to get the most-recent page.
    Returns: { items: [...], has_more: bool }
    """
    query: dict = {"conversation_id": ObjectId(conversation_id)}
    if cursor:
        query["_id"] = {"$lt": ObjectId(cursor)}

    docs = list(
        _messages()
        .find(query)
        .sort("_id", DESCENDING)
        .limit(page_size + 1)
    )

    has_more = len(docs) > page_size
    if has_more:
        docs.pop()

    items = [_doc_to_dto(d) for d in docs]
    return {"items": items, "has_more": has_more}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _doc_to_dto(doc: dict) -> dict:
    """Convert a MongoDB document to a serialisable dict for the response schema."""
    return {
        "id": str(doc["_id"]),
        "role": doc.get("role", ""),
        "content": doc.get("content", ""),
        "content_id": doc.get("content_id"),
        "content_type": doc.get("content_type"),
        "sources": doc.get("sources"),
        "created_at": doc.get("created_at", datetime.now(timezone.utc)).isoformat(),
    }
