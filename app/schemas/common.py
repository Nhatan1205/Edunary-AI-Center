"""Common shared schemas."""

from pydantic import BaseModel
from typing import Optional, Any


class APIResponse(BaseModel):
    """Standard API response wrapper."""

    success: bool = True
    message: str = ""
    data: Optional[Any] = None
