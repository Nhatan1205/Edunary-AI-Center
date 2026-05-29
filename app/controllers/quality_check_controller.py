from fastapi import APIRouter, Depends, HTTPException

from app.auth.dependencies import verify_request
from app.core.logging import get_logger
from app.schemas.quality_check_schema import QualityCheckRequest, QualityCheckResponse
from app.services.quality_check_service import run_quality_check

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(verify_request)])


@router.post("/analyze", response_model=QualityCheckResponse)
async def analyze_quality_endpoint(request: QualityCheckRequest):
    logger.info(
        f"AI Quality Check request: course_id={request.course_id}, "
        f"course_title='{request.course_title}', "
        f"curriculum_sections={len(request.curriculum)}"
    )

    try:
        response_data = await run_quality_check(request)
    except ValueError as e:
        logger.error(f"AI Quality Check failed (validation): {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"AI Quality Check failed (unexpected): {e}")
        raise HTTPException(status_code=500, detail="Internal AI error during quality check")

    return response_data
