"""
Transcription & Translation schemas.

Two operation modes:
  1. generate  — Whisper STT: video_file_url → vtt_content + detected_language
  2. translate — LLM:         source_vtt_content + target_language → translated vtt_content
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field

from app.schemas.quiz_schema import STTConfig
from app.schemas.model_config import ModelConfig


# ─── Generate (Whisper STT) ───────────────────────────────────────────────────

class GenerateCaptionRequest(BaseModel):
    """
    Request body for POST /api/transcription/generate.
    Sent by CaptionGenerationJobService (.NET Infrastructure).
    """
    video_file_url: str = Field(..., description="Publicly accessible video/HLS URL")
    stt_config: Optional[STTConfig] = None


class GenerateCaptionData(BaseModel):
    vtt_content: str            # Full WebVTT string (WEBVTT header + cues)
    detected_language: str      # Whisper lowercase name e.g. "english", "vietnamese"
    duration_seconds: float     # Total audio duration in seconds


class GenerateCaptionResponse(BaseModel):
    success: bool = True
    data: Optional[GenerateCaptionData] = None
    message: str = ""


# ─── Translate (LLM) ──────────────────────────────────────────────────────────

class TranslateCaptionRequest(BaseModel):
    """
    Request body for POST /api/transcription/translate.
    Sent by CaptionGenerationJobService when source transcript already exists.
    """
    source_vtt_content: str         # Raw VTT text from the source transcript
    source_language: str            # e.g. "english"
    target_language: str            # e.g. "vietnamese"
    llm_config: Optional[ModelConfig] = None


class TranslateCaptionData(BaseModel):
    vtt_content: str                # Translated WebVTT string
    target_language: str            # Confirmed target language


class TranslateCaptionResponse(BaseModel):
    success: bool = True
    data: Optional[TranslateCaptionData] = None
    message: str = ""
