"""
Pydantic schemas for the User Embedding endpoints.
Called by the .NET backend UserEmbeddingJobService via HMAC-authenticated POST.
"""

from pydantic import BaseModel
from typing import Optional
from app.schemas.rag_schema import EmbeddingConfig, QdrantConfig


# ── Input Models ─────────────────────────────────────────────────────────────

class UserProfileData(BaseModel):
    """
    Full data for a single user to be embedded.
    Stored as Qdrant payload so user info can be retrieved without a DB roundtrip.
    """
    user_id: str                                    # ASP.NET Identity GUID string
    user_name: str
    full_name: Optional[str] = None
    goal: Optional[str] = None
    skill_level: Optional[str] = None
    preferred_categories: Optional[list[str]] = None    # resolved names, not IDs
    preferred_topics: Optional[list[str]] = None        # resolved names, not IDs
    weekly_hours: Optional[int] = None


class EmbedUserProfileRequest(BaseModel):
    """Request body for POST /api/user-embeddings/embed — single user."""
    user_profile: UserProfileData
    embedding_config: Optional[EmbeddingConfig] = None
    qdrant_config: Optional[QdrantConfig] = None


class BatchEmbedUsersRequest(BaseModel):
    """Request body for POST /api/user-embeddings/embed-batch — multiple users."""
    user_profiles: list[UserProfileData]
    embedding_config: Optional[EmbeddingConfig] = None
    qdrant_config: Optional[QdrantConfig] = None


class SyncStatusRequest(BaseModel):
    """
    Request body for POST /api/user-embeddings/sync-status.
    Compares provided user IDs vs what is actually embedded in Qdrant.
    """
    user_ids: list[str]
    qdrant_config: Optional[QdrantConfig] = None


# ── Output Models ─────────────────────────────────────────────────────────────

class EmbedUserProfileResponse(BaseModel):
    success: bool
    user_id: str
    message: str


class BatchEmbedUsersResponse(BaseModel):
    success: bool
    total_embedded: int
    failed_ids: list[str]
    message: str


class SyncStatusResponse(BaseModel):
    embedded_ids: list[str]
    missing_ids: list[str]
    total_embedded: int
    total_missing: int
