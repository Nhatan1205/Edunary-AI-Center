from pydantic import BaseModel
from typing import Dict, Optional, List

class CourseCard(BaseModel):
    id: int
    title: str
    subtitle: Optional[str] = None
    price: float
    imageUrl: Optional[str] = None
    level: str
    ratings: Optional[float] = None
    totalStudents: Optional[int] = None
    instructorName: str
    courseUrl: str

class ChatResponse(BaseModel):
    session_id: Optional[str] = None   
    reply: str
    messageType: str = "text"
    courses: Optional[List[CourseCard]] = None        