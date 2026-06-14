from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel, Field

from app.schemas.model_config import ModelConfig
from app.schemas.rag_schema import EmbeddingConfig, QdrantConfig


class CurriculumItem(BaseModel):
    title: str
    content_type: str  # article, video, quiz, assignment


class CurriculumSection(BaseModel):
    section_title: str
    learning_objectives: str = ""
    items: List[CurriculumItem] = Field(default_factory=list)


class ContentBatch(BaseModel):
    batch_index: int
    sections: List[str]
    content_text: str
    # Item IDs in this batch — used to exclude them from cross-section vector search
    item_ids: List[str] = Field(default_factory=list)


class QualityCheckRequest(BaseModel):
    course_id: int
    course_title: str
    course_subtitle: str = ""
    course_description: str = ""
    course_category: str = "None"
    course_topics: List[str] = Field(default_factory=list)
    learning_objectives: List[str] = Field(default_factory=list)
    requirements: List[str] = Field(default_factory=list)
    target_audience: List[str] = Field(default_factory=list)
    curriculum: List[CurriculumSection] = Field(default_factory=list)
    content_batches: List[ContentBatch] = Field(default_factory=list)
    check_mode: Optional[str] = "full"
    changed_fields: Optional[List[str]] = Field(default_factory=list)
    llm_config: Optional[ModelConfig] = None
    validate_llm_config: Optional[ModelConfig] = None
    # Optional configs for cross-section context injection via Qdrant
    embedding_config: Optional[EmbeddingConfig] = None
    qdrant_config: Optional[QdrantConfig] = None


class AiCheckIssue(BaseModel):
    ruleId: str
    category: str
    severity: str
    location: str
    description: str
    evidence: str = ""
    suggestion: str = ""


class QualityCheckResponse(BaseModel):
    issues: List[AiCheckIssue] = Field(default_factory=list)
    analysis_summary: str = ""
