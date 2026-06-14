"""
Pydantic schemas for the Course Curriculum Embedding endpoints.
Called by the .NET backend CourseCurriculumEmbeddingJobService.
"""

from pydantic import BaseModel
from typing import Optional, List
from app.schemas.rag_schema import EmbeddingConfig, QdrantConfig


# ── Input Models ─────────────────────────────────────────────────────────────

class CurriculumCourseMetadata(BaseModel):
    """Metadata of the course for context filtering/enrichment."""
    course_title: str
    course_category: Optional[str] = None
    course_topics: Optional[List[str]] = None
    updated_at: Optional[str] = None


class CurriculumLectureData(BaseModel):
    """
    Full data for a single lecture (article or video text) to be chunked/embedded.
    """
    section_id: str
    section_index: int
    section_title: str
    section_learning_objectives: Optional[str] = None
    item_id: str
    item_index: int
    item_title: str
    item_type: str  # 'article' or 'video'
    content_text: str
    word_count: int
    has_code_blocks: bool
    preceding_item_ids: List[str]


class CurriculumEmbedRequest(BaseModel):
    """Request body for POST /api/curriculum/embed."""
    course_id: int
    course_metadata: CurriculumCourseMetadata
    lectures: List[CurriculumLectureData]
    embedding_config: Optional[EmbeddingConfig] = None
    qdrant_config: Optional[QdrantConfig] = None


class DeleteCurriculumEmbeddingRequest(BaseModel):
    """Request body for DELETE /api/curriculum/{course_id}."""
    course_id: int
    qdrant_config: Optional[QdrantConfig] = None


# ── Output Models ─────────────────────────────────────────────────────────────

class CurriculumEmbedResponse(BaseModel):
    success: bool
    course_id: int
    total_chunks: int
    message: str


class DeleteCurriculumEmbeddingResponse(BaseModel):
    success: bool
    course_id: int
    message: str


class CurriculumSearchResult(BaseModel):
    """A single search result from the curriculum collection."""
    course_id: int
    section_id: str
    section_title: str
    item_id: str
    item_title: str
    item_type: str
    content_text: str
    score: float
