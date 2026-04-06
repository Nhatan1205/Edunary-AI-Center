"""Chat-related Pydantic schemas (request / response models)."""

from pydantic import BaseModel
from typing import Optional, List


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""

    session_id: Optional[str] = None
    message: str


class CourseCard(BaseModel):
    """Course card data returned in chat responses."""

    id: int
    title: str
    subtitle: Optional[str] = None
    imageUrl: Optional[str] = None
    instructorName: str
    courseUrl: str


class ChatResponse(BaseModel):
    """Response body for the chat endpoint."""

    session_id: Optional[str] = None
    reply: str
    messageType: str = "text"
    courses: Optional[List[CourseCard]] = None
