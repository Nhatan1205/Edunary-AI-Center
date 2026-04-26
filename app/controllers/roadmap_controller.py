"""
Roadmap controller — AI roadmap generation endpoint.

Called by the .NET backend (authenticated via X-API-Key).
The backend provides the topic, level, and optional model_config
(read from SystemSettings), so this endpoint never reads .env for LLM config.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional

from app.auth.api_key import verify_api_key
from app.schemas.model_config import ModelConfig
from app.schemas.common import APIResponse
from app.services.roadmap_service import generate_roadmap
from app.core.logging import get_logger

logger = get_logger(__name__)

# All routes in this router require X-API-Key authentication
router = APIRouter(dependencies=[Depends(verify_api_key)])


class GenerateRoadmapRequest(BaseModel):
    """Request body sent by the .NET backend."""

    topic: str
    level: str = "Beginner"  # Beginner | Intermediate | Advanced | AllLevel
    llm_config: Optional[ModelConfig] = None  # Renamed from model_config — Pydantic V2 reserves that name


@router.post("/generate", response_model=APIResponse)
async def roadmap_endpoint(request: GenerateRoadmapRequest):
    """
    Generate an AI learning roadmap for a given topic and level.

    Returns a graph structure with nodes (learning topics) and edges (prerequisites).
    Nodes contain label + description rather than real CourseIds — the .NET backend
    or frontend can later map each node to an actual course in the DB.

    Auth: X-API-Key header (set AUTH_ENABLED=true in production).
    """
    logger.info(f"Generating roadmap: topic='{request.topic}', level='{request.level}'")

    llm_config_dict = request.llm_config.model_dump() if request.llm_config else None

    data = generate_roadmap(topic=request.topic,level=request.level,model_config=llm_config_dict)

    return APIResponse(
        success=True,
        message="Roadmap generated successfully",
        data=data,
    )
