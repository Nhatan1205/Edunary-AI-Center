"""
Course Assistant Schemas — Pydantic models for request/response.
Completely independent from the existing chat pipeline.
"""

from typing import Optional, List
from pydantic import BaseModel, Field

from app.schemas.model_config import ModelConfig


# ── Request Models ────────────────────────────────────────────────────────────

class CourseAssistantCourseData(BaseModel):
    """Metadata of the course for system prompting/context."""
    title: str = Field(..., description="Course title for system prompt context")
    subtitle: Optional[str] = Field(default="", description="Course subtitle")
    description: Optional[str] = Field(default="", description="Course description")
    learning_objectives: Optional[str] = Field(default="", description="Course learning objectives")
    requirements: Optional[str] = Field(default="", description="Course requirements")
    target_audience: Optional[str] = Field(default="", description="Who the course is for")


class CourseAssistantChatRequest(BaseModel):
    """
    Payload sent from .NET backend to AI Center when a student sends a message.
    The .NET backend has already verified enrollment before forwarding here.
    """
    user_id: str = Field(..., description="ASP.NET Identity user GUID")
    course_id: int = Field(..., description="The course the student is enrolled in")
    course_data: CourseAssistantCourseData = Field(..., description="Course metadata DTO")
    content_id: Optional[str] = Field(default=None, description="Current lecture/quiz/assignment item_id")
    content_type: Optional[str] = Field(default="lecture", description="lecture | quiz | assignment")
    media_type: Optional[str] = Field(default=None, description="video | article | empty")
    content_title: Optional[str] = Field(default=None, description="Title of the current content item")
    message: str = Field(..., description="The student's question")
    llm_config: Optional[ModelConfig] = None



class CourseAssistantHistoryRequest(BaseModel):
    """Cursor-paginated history request."""
    user_id: str
    course_id: int
    cursor: Optional[str] = Field(default=None, description="MongoDB ObjectId string of the oldest message in current view")
    page_size: int = Field(default=20, ge=1, le=50)


class CourseAssistantClearRequest(BaseModel):
    """Request to clear a student's conversation for a course."""
    user_id: str
    course_id: int


# ── Response Models ───────────────────────────────────────────────────────────

class CourseAssistantMessageDto(BaseModel):
    """A single message in the conversation."""
    id: str
    role: str                           # "user" | "assistant"
    content: str
    content_id: Optional[str] = None
    content_type: Optional[str] = None
    sources: Optional[List[str]] = None
    created_at: str


class CourseAssistantChatResponse(BaseModel):
    """Response returned after processing a student's message."""
    reply: str
    message_type: str = "text"          # "text" | "quiz"
    sources: Optional[List[str]] = None


class CourseAssistantHistoryResponse(BaseModel):
    """Cursor-paginated list of messages (newest-first)."""
    items: List[CourseAssistantMessageDto]
    has_more: bool
