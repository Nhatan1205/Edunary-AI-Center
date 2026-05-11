from fastapi import APIRouter, Depends, HTTPException

from app.auth.dependencies import verify_request
from app.core.logging import get_logger
from app.schemas.quiz_schema import GenerateQuizRequest, GenerateQuizResponse
from app.services.quiz_service import generate_quiz

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(verify_request)])


@router.post("/generate", response_model=GenerateQuizResponse)
async def generate_quiz_endpoint(request: GenerateQuizRequest):
    logger.info(
        f"Quiz generation request: lecture='{request.lecture_title}', "
        f"course='{request.course_title}', "
        f"num_questions={request.quiz_config.num_questions}, "
        f"difficulty={request.quiz_config.difficulty}, "
        f"types={request.quiz_config.question_types}, "
        f"has_content={bool(request.lecture_content)}, "
        f"has_video_url={bool(request.video_file_url)}"
    )

    try:
        quiz_data = generate_quiz(request)
    except ValueError as e:
        logger.error(f"Quiz generation failed (validation): {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Quiz generation failed (unexpected): {e}")
        raise HTTPException(status_code=500, detail="Internal AI error during quiz generation")

    return GenerateQuizResponse(
        success=True,
        data=quiz_data,
        message=f"Successfully generated {len(quiz_data.questions)} questions",
    )
