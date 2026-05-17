"""Chat-related Pydantic schemas (request / response models)."""

from pydantic import BaseModel
from typing import Optional, List


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""

    session_id: Optional[str] = None
    message: str
    user_id: Optional[str] = None  # ASP.NET Identity GUID — omit for anonymous


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
    is_personalized: bool = False  # True when response used authenticated user's profile
