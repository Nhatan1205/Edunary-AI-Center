"""
Quiz Generator schemas.

The .NET backend sends lecture content (text for articles, or video_file_url for
videos without captions) together with learning objectives, quiz configuration,
and the standard llm_config + stt_config blocks read from SystemSettings.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field

from app.schemas.model_config import ModelConfig


# ─── STT Config ───────────────────────────────────────────────────────────────

class STTConfig(BaseModel):
    """
    Speech-to-Text provider config sent by the .NET backend (from SystemSettings).
    Only needed when video_file_url is provided and no caption text is available.
    """
    api_key: str = ""
    model_name: str = "whisper-1"


# ─── Learning Objectives ──────────────────────────────────────────────────────

class LearningObjectives(BaseModel):
    """Dual-level pedagogical anchors extracted by the .NET backend."""
    course_level: str = ""   # Course.LearningObjectives
    section_level: str = ""  # SectionSchema.LearningObjectives


# ─── Quiz Configuration ───────────────────────────────────────────────────────

class QuizConfig(BaseModel):
    num_questions: int = Field(default=5, ge=1, le=20)
    question_types: list[str] = Field(
        default=["SingleChoice", "MultipleChoice", "TrueFalse"]
    )
    difficulty: str = "Medium"          # Easy | Medium | Hard
    prompt_description: str = ""        # Instructor freeform directives


# ─── Request ──────────────────────────────────────────────────────────────────

class GenerateQuizRequest(BaseModel):
    """
    Request body sent by QuizGenerationJobService (.NET Infrastructure).

    Content source (exactly one should be non-empty):
      - lecture_content : Plain text already extracted (.NET stripped HTML or parsed VTT)
      - video_file_url  : Direct video URL — AI Center runs FFmpeg + Whisper STT
    """
    lecture_content: str = ""           # Article text OR pre-parsed VTT caption text
    video_file_url: str = ""            # Set only when video has NO existing captions
    lecture_title: str = ""
    course_title: str = ""
    learning_objectives: LearningObjectives = Field(default_factory=LearningObjectives)
    quiz_config: QuizConfig = Field(default_factory=QuizConfig)
    llm_config: Optional[ModelConfig] = None
    stt_config: Optional[STTConfig] = None


# ─── Response ─────────────────────────────────────────────────────────────────

class GeneratedChoice(BaseModel):
    text: str
    is_correct: bool
    sort_order: int


class GeneratedQuestion(BaseModel):
    name: str                           # Question text
    type: str                           # SingleChoice | MultipleChoice | TrueFalse
    explanation: str = ""               # Shown after answering
    sort_order: int
    choices: list[GeneratedChoice]


class GenerateQuizData(BaseModel):
    questions: list[GeneratedQuestion]
    content_summary: str = ""           # Brief summary of analyzed content


class GenerateQuizResponse(BaseModel):
    success: bool = True
    data: Optional[GenerateQuizData] = None
    message: str = ""
