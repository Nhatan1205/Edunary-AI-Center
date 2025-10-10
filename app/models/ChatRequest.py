from pydantic import BaseModel
from typing import Dict, Optional

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str