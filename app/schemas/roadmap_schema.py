
from pydantic import BaseModel
from typing import Optional
from app.schemas.model_config import ModelConfig


# ── Request schemas ────────────────────────────────────────────────────────────

class StudentProfile(BaseModel):
    skill_level: str
    goal: Optional[str] = None
    preferred_topics: list[str] = []
    preferred_categories: list[str] = []
    weekly_hours: Optional[int] = None
    enrolled_course_ids: list[int] = []


class RoadmapData(BaseModel):
    description: str
    roadmap_topic: str


class GeneratePersonalizedRoadmapRequest(BaseModel):
    """
    Request to generate a personalized roadmap.
    courses_catalog is no longer sent from Edunary backend —
    AI Center fetches relevant courses via Qdrant semantic search.
    """
    student_profile: StudentProfile
    roadmap_data: RoadmapData
    llm_config: Optional[ModelConfig] = None


# ── Response schemas ───────────────────────────────────────────────────────────

class RoadmapNode(BaseModel):
    clientNodeId: str
    courseId: int
    reason: str
    positionX: float = 250.0
    positionY: float = 0.0
    sortOrder: int


class RoadmapEdge(BaseModel):
    sourceNodeId: str
    targetNodeId: str


class PersonalizedRoadmapData(BaseModel):
    nodes: list[RoadmapNode]
    edges: list[RoadmapEdge]
    summary: str
    searched_course_ids: list[int] = []
    """IDs of all courses retrieved from Qdrant search — used by Edunary backend to validate node courseIds."""


class PersonalizedRoadmapResponse(BaseModel):
    success: bool = True
    data: Optional[PersonalizedRoadmapData] = None
    message: str = ""
