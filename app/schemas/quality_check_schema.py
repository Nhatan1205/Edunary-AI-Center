from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel, Field

from app.schemas.model_config import ModelConfig


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


class QualityCheckRequest(BaseModel):
    course_id: int
    course_title: str
    course_subtitle: str = ""
    course_description: str = ""
    learning_objectives: List[str] = Field(default_factory=list)
    requirements: List[str] = Field(default_factory=list)
    target_audience: List[str] = Field(default_factory=list)
    curriculum: List[CurriculumSection] = Field(default_factory=list)
    content_batches: List[ContentBatch] = Field(default_factory=list)
    llm_config: Optional[ModelConfig] = None


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
