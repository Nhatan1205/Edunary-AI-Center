from fastapi import APIRouter, Depends, HTTPException

from app.auth.dependencies import verify_request
from app.schemas.roadmap_schema import (
    GeneratePersonalizedRoadmapRequest,
    PersonalizedRoadmapResponse,
)
from app.core.logging import get_logger
from app.services.roadmap_service import generate_roadmap

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(verify_request)])


@router.post("/generate", response_model=PersonalizedRoadmapResponse)
async def generate_personalized_roadmap_endpoint(
    request: GeneratePersonalizedRoadmapRequest,
):
    profile = request.student_profile
    logger.info(
        f"Personalized roadmap request: topic='{request.roadmap_data.roadmap_topic}', "
        f"level='{profile.skill_level}', goal='{profile.goal}'"
    )

    try:
        roadmap_data = generate_roadmap(request)
    except ValueError as e:
        logger.error(f"Personalized roadmap generation failed (schema/JSON): {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Personalized roadmap generation failed (unexpected): {e}")
        raise HTTPException(status_code=500, detail="Internal AI error")

    return PersonalizedRoadmapResponse(
        success=True,
        data=roadmap_data,
        message="Personalized roadmap generated successfully",
    )
