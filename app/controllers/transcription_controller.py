"""
Transcription controller.

Exposes two HMAC-protected endpoints for the .NET CaptionGenerationJobService:

  POST /api/transcription/generate   — Whisper STT → WebVTT + detected language
  POST /api/transcription/translate  — LLM translate VTT to a new language
"""

from fastapi import APIRouter, Depends, HTTPException

from app.auth.dependencies import verify_request
from app.core.logging import get_logger
from app.schemas.transcription_schema import (
    GenerateCaptionRequest,
    GenerateCaptionResponse,
    GenerateCaptionData,
    TranslateCaptionRequest,
    TranslateCaptionResponse,
    TranslateCaptionData,
)
from app.services.transcription_service import generate_caption_vtt, translate_caption_vtt

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(verify_request)])


@router.post("/generate", response_model=GenerateCaptionResponse)
async def generate_caption_endpoint(request: GenerateCaptionRequest):
    """
    Transcribe a video to WebVTT using Whisper STT.

    Handles long videos via chunked transcription (20-min chunks).
    Returns the full VTT string, detected language, and audio duration.
    """
    logger.info(f"[Transcription] Generate caption request: url={request.video_file_url[:80]}...")

    try:
        vtt_content, detected_language, duration_seconds = generate_caption_vtt(
            video_file_url=request.video_file_url,
            stt_config=request.stt_config,
        )
    except ValueError as e:
        logger.error(f"[Transcription] Generate failed (config): {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"[Transcription] Generate failed (unexpected): {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    logger.info(
        f"[Transcription] Generated VTT: language={detected_language}, "
        f"duration={duration_seconds:.1f}s, size={len(vtt_content)} chars"
    )

    return GenerateCaptionResponse(
        success=True,
        data=GenerateCaptionData(
            vtt_content=vtt_content,
            detected_language=detected_language,
            duration_seconds=duration_seconds,
        ),
        message=f"Caption generated successfully ({detected_language})",
    )


@router.post("/translate", response_model=TranslateCaptionResponse)
async def translate_caption_endpoint(request: TranslateCaptionRequest):
    """
    Translate an existing WebVTT caption to a new language using the LLM.

    Used when a source transcript already exists for the video —
    avoids re-calling Whisper STT.
    """
    logger.info(
        f"[Transcription] Translate caption request: "
        f"{request.source_language} → {request.target_language}, "
        f"source_size={len(request.source_vtt_content)} chars"
    )

    try:
        translated_vtt = translate_caption_vtt(
            source_vtt_content=request.source_vtt_content,
            source_language=request.source_language,
            target_language=request.target_language,
            llm_config=request.llm_config.model_dump() if request.llm_config else None,
        )
    except Exception as e:
        logger.error(f"[Transcription] Translate failed: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

    logger.info(f"[Transcription] Translated VTT: {len(translated_vtt)} chars")

    return TranslateCaptionResponse(
        success=True,
        data=TranslateCaptionData(
            vtt_content=translated_vtt,
            target_language=request.target_language,
        ),
        message=f"Caption translated to {request.target_language} successfully",
    )
