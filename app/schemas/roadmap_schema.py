

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


class CourseCatalogItem(BaseModel):
    id: int
    title: str
    level: str
    topics: list[str] = []
    category: Optional[str] = None
    ratings: float = 0.0
    total_students: int = 0
    learning_objectives: Optional[str] = None


class GeneratePersonalizedRoadmapRequest(BaseModel):
    student_profile: StudentProfile
    roadmap_data: RoadmapData
    courses_catalog: list[CourseCatalogItem]
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


class PersonalizedRoadmapResponse(BaseModel):
    success: bool = True
    data: Optional[PersonalizedRoadmapData] = None
    message: str = ""
