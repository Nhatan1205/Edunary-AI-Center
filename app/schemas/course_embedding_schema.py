"""
Pydantic schemas for the Course Embedding endpoints.
Called by the .NET backend CourseEmbeddingJobService via HMAC-authenticated POST.
"""

from pydantic import BaseModel
from typing import Optional
from app.schemas.rag_schema import EmbeddingConfig, QdrantConfig


# ── Input Models ─────────────────────────────────────────────────────────────

class CourseData(BaseModel):
    """
    Full data for a single course to be embedded.
    Stored as Qdrant payload so course info can be retrieved without DB roundtrip.
    """
    course_id: int
    title: str
    subtitle: Optional[str] = None
    description: Optional[str] = None
    learning_objectives: Optional[str] = None   # JSON array string
    requirements: Optional[str] = None          # JSON array string
    target_audience: Optional[str] = None       # JSON array string
    category_name: Optional[str] = None
    topics: Optional[list[str]] = None
    level: Optional[str] = None
    price: Optional[float] = None
    image_url: Optional[str] = None
    instructor_name: Optional[str] = None


class EmbedCourseRequest(BaseModel):
    """Request body for POST /api/course-embeddings/embed — single course."""
    course: CourseData
    embedding_config: Optional[EmbeddingConfig] = None
    qdrant_config: Optional[QdrantConfig] = None


class BatchEmbedCoursesRequest(BaseModel):
    """Request body for POST /api/course-embeddings/embed-batch — multiple courses."""
    courses: list[CourseData]
    embedding_config: Optional[EmbeddingConfig] = None
    qdrant_config: Optional[QdrantConfig] = None


class DeleteCourseEmbeddingRequest(BaseModel):
    """Request body for POST /api/course-embeddings/delete — remove a single course point."""
    course_id: int
    qdrant_config: Optional[QdrantConfig] = None


class SyncStatusRequest(BaseModel):
    """
    Request body for POST /api/course-embeddings/sync-status.
    Compares provided course IDs vs what is actually embedded in Qdrant.
    """
    course_ids: list[int]
    qdrant_config: Optional[QdrantConfig] = None


class SearchCoursesRequest(BaseModel):
    """Request body for POST /api/course-embeddings/search — semantic search."""
    query: str
    limit: int = 5
    embedding_config: Optional[EmbeddingConfig] = None
    qdrant_config: Optional[QdrantConfig] = None


# ── Output Models ─────────────────────────────────────────────────────────────

class EmbedCourseResponse(BaseModel):
    success: bool
    course_id: int
    message: str


class BatchEmbedCoursesResponse(BaseModel):
    success: bool
    total_embedded: int
    failed_ids: list[int]
    message: str


class DeleteCourseEmbeddingResponse(BaseModel):
    success: bool
    course_id: int
    message: str


class SyncStatusResponse(BaseModel):
    embedded_ids: list[int]
    missing_ids: list[int]
    total_embedded: int
    total_missing: int


class CourseSearchResult(BaseModel):
    """A single search result returned by semantic search."""
    course_id: int
    title: str
    subtitle: Optional[str] = None
    image_url: Optional[str] = None
    instructor_name: Optional[str] = None
    price: Optional[float] = None
    level: Optional[str] = None
    category_name: Optional[str] = None
    score: float


class SearchCoursesResponse(BaseModel):
    results: list[CourseSearchResult]
    total: int
