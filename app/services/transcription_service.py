"""
Transcription service.

Converts lecture content to plain text for the quiz generation pipeline.

Content source priority:
  1. lecture_content provided directly (article HTML already stripped by .NET,
     OR VTT caption already parsed by .NET) → passthrough, no STT needed.
  2. video_file_url provided → download audio via ffmpeg → Whisper STT.

This service is the foundation for the future AI-generated captions feature.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import os
import httpx

from app.core.logging import get_logger
from app.schemas.quiz_schema import STTConfig

logger = get_logger(__name__)

# Maximum characters to send to the LLM (prevent token overflow)
MAX_CONTENT_CHARS = 12_000


def get_lecture_text(
    lecture_content: str,
    video_file_url: str,
    stt_config: STTConfig | None,
) -> str:
    """
    Return plain lecture text ready for the quiz generation pipeline.

    Args:
        lecture_content : Pre-extracted text from .NET (article or VTT). May be empty.
        video_file_url  : Video URL to transcribe via Whisper. Used only when
                          lecture_content is empty (no captions available).
        stt_config      : Whisper API credentials from SystemSettings.

    Returns:
        Plain text string, truncated to MAX_CONTENT_CHARS.

    Raises:
        ValueError: If neither source yields meaningful text.
    """
    # --- Path 1: Text already provided (.NET extracted article or parsed VTT) ---
    if lecture_content and lecture_content.strip():
        text = lecture_content.strip()
        logger.info(f"[Transcription] Using pre-extracted text ({len(text)} chars)")
        return _truncate(text)

    # --- Path 2: Video without captions → Whisper STT ---
    if video_file_url and video_file_url.strip():
        logger.info(f"[Transcription] No caption text — running Whisper STT on video: {video_file_url}")
        text = _transcribe_video(video_file_url, stt_config)
        return _truncate(text)

    raise ValueError(
        "No lecture content available. "
        "Provide either article content, VTT captions, or a video URL."
    )


def _transcribe_video(video_url: str, stt_config: STTConfig | None) -> str:
    """
    Download audio from video URL via FFmpeg, then transcribe with OpenAI Whisper API.

    Args:
        video_url  : Publicly accessible video/HLS URL.
        stt_config : Whisper API key + model name from SystemSettings.

    Returns:
        Transcription text.
    """
    api_key = (stt_config.api_key if stt_config else "") or _get_fallback_key()
    model_name = (stt_config.model_name if stt_config else "") or "whisper-1"

    if not api_key:
        raise ValueError(
            "STT API key is not configured. "
            "Set STT_ApiKey in Edunary Admin → System Settings."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")

        # Extract audio via FFmpeg (handles HLS, MP4, WebM, etc.)
        logger.info(f"[STT] Extracting audio from: {video_url}")
        _extract_audio_ffmpeg(video_url, audio_path)

        # Transcribe with OpenAI Whisper API
        logger.info(f"[STT] Transcribing with model={model_name}")
        transcript = _call_whisper_api(audio_path, api_key, model_name)

    logger.info(f"[STT] Transcript length: {len(transcript)} chars")
    return transcript


def _extract_audio_ffmpeg(video_url: str, output_path: str) -> None:
    """Use FFmpeg to extract audio as MP3 from a video URL."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_url,
        "-vn",                  # No video
        "-ar", "16000",         # 16kHz sample rate (Whisper optimal)
        "-ac", "1",             # Mono
        "-b:a", "64k",          # 64kbps — small file, good quality for speech
        output_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,        # 5 minute timeout for long videos
        )
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr[-500:]}")
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg is not installed or not in PATH. "
            "Install it to enable video transcription."
        )


def _call_whisper_api(audio_path: str, api_key: str, model_name: str) -> str:
    """Call OpenAI Whisper API to transcribe the audio file."""
    with open(audio_path, "rb") as audio_file:
        response = httpx.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            data={"model": model_name, "response_format": "text"},
            files={"file": ("audio.mp3", audio_file, "audio/mpeg")},
            timeout=180.0,
        )
    if response.status_code != 200:
        raise RuntimeError(
            f"Whisper API error {response.status_code}: {response.text[:300]}"
        )
    return response.text.strip()


def _get_fallback_key() -> str:
    """Fallback to .env OPENAI_KEY if no stt_config key provided."""
    from app.core.config import get_settings
    return getattr(get_settings(), "OPENAI_KEY", "")


def _truncate(text: str) -> str:
    """Truncate text to MAX_CONTENT_CHARS with a notice."""
    if len(text) <= MAX_CONTENT_CHARS:
        return text
    logger.warning(
        f"[Transcription] Content truncated from {len(text)} to {MAX_CONTENT_CHARS} chars"
    )
    return text[:MAX_CONTENT_CHARS] + "\n\n[Content truncated for token limit]"
