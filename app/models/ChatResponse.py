from pydantic import BaseModel
from typing import Dict, Optional

class ChatResponse(BaseModel):
    session_id: Optional[str] = None   
    reply: str        