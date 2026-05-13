"""
Transcription service.

Converts lecture content to plain text for the quiz generation pipeline,
and generates/translates WebVTT caption files for the AI caption feature.

Content source priority (quiz pipeline):
  1. lecture_content provided directly (article HTML already stripped by .NET,
     OR VTT caption already parsed by .NET) → passthrough, no STT needed.
  2. video_file_url provided → download audio via ffmpeg → Whisper STT.

Caption generation (AI caption pipeline):
  - generate_caption_vtt()  : Whisper STT → full WebVTT with timestamps
  - translate_caption_vtt() : LLM translate source VTT → target language VTT
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import os
import math
import httpx

from app.core.logging import get_logger
from app.schemas.quiz_schema import STTConfig

logger = get_logger(__name__)

# Maximum characters to send to the LLM (prevent token overflow)
MAX_CONTENT_CHARS = 12_000

# Chunk duration for long-video transcription (20 minutes)
_CHUNK_DURATION_SECS = 20 * 60

# Segments per LLM batch for translation
_TRANSLATION_BATCH_SIZE = 20


# ─── Quiz pipeline ────────────────────────────────────────────────────────────

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
        text = _transcribe_video_plain(video_file_url, stt_config)
        return _truncate(text)

    raise ValueError(
        "No lecture content available. "
        "Provide either article content, VTT captions, or a video URL."
    )


# ─── Caption generation pipeline ──────────────────────────────────────────────

def generate_caption_vtt(
    video_file_url: str,
    stt_config: STTConfig | None,
) -> tuple[str, str, float]:
    """
    Transcribe a video using Whisper STT and return a WebVTT caption string.

    Handles long videos via chunked transcription (20-min chunks) so the
    audio file never exceeds Whisper's 25 MB upload limit.

    Args:
        video_file_url : Publicly accessible video/HLS URL.
        stt_config     : Whisper API credentials.

    Returns:
        (vtt_content, detected_language, duration_seconds)
        - vtt_content       : Full WebVTT string ready to save as .vtt file.
        - detected_language : Whisper-detected language name, e.g. "english".
        - duration_seconds  : Total audio duration.

    Raises:
        ValueError : Missing STT API key.
        RuntimeError: FFmpeg or Whisper API failure.
    """
    api_key = (stt_config.api_key if stt_config else "") or _get_fallback_key()
    model_name = (stt_config.model_name if stt_config else "") or "whisper-1"

    if not api_key:
        raise ValueError(
            "STT API key is not configured. "
            "Set STT_ApiKey in Edunary Admin → System Settings."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        full_audio_path = os.path.join(tmpdir, "full_audio.mp3")

        # 1. Extract full audio
        logger.info(f"[Caption] Extracting audio from: {video_file_url}")
        _extract_audio_ffmpeg(video_file_url, full_audio_path)

        # 2. Get total duration
        total_duration = _get_audio_duration(full_audio_path)
        logger.info(f"[Caption] Audio duration: {total_duration:.1f}s")

        # 3. Chunk + transcribe
        num_chunks = math.ceil(total_duration / _CHUNK_DURATION_SECS)
        all_segments: list[dict] = []
        detected_language = "english"

        for chunk_idx in range(num_chunks):
            start_sec = chunk_idx * _CHUNK_DURATION_SECS
            chunk_path = os.path.join(tmpdir, f"chunk_{chunk_idx}.mp3")

            logger.info(f"[Caption] Chunk {chunk_idx + 1}/{num_chunks}: {start_sec}s → {start_sec + _CHUNK_DURATION_SECS}s")
            _extract_audio_chunk(full_audio_path, chunk_path, start_sec, _CHUNK_DURATION_SECS)

            chunk_data = _call_whisper_api_verbose(chunk_path, api_key, model_name)

            if chunk_idx == 0:
                detected_language = chunk_data.get("language", "english")

            # Offset segment timestamps by chunk start
            for seg in chunk_data.get("segments", []):
                all_segments.append({
                    "start": seg["start"] + start_sec,
                    "end":   seg["end"]   + start_sec,
                    "text":  seg["text"].strip(),
                })

        logger.info(f"[Caption] Total segments: {len(all_segments)}, language: {detected_language}")

    # 4. Build VTT
    vtt_content = _build_vtt(all_segments)
    return vtt_content, detected_language, total_duration


def translate_caption_vtt(
    source_vtt_content: str,
    source_language: str,
    target_language: str,
    llm_config=None,
) -> str:
    """
    Translate a WebVTT caption file to a new language using the LLM.

    Whisper is NOT called — translation uses only the LLM to keep costs low
    when adding additional caption languages to a video that already has a
    source transcript.

    Args:
        source_vtt_content : Original VTT string (from Whisper source transcript).
        source_language    : Source language name, e.g. "english".
        target_language    : Target language name, e.g. "vietnamese".
        llm_config         : Optional backend-provided LLM config dict.

    Returns:
        Translated WebVTT string.
    """
    from app.llm.factory import create_llm, create_llm_from_config
    from app.prompts.caption_prompts import TRANSLATION_SYSTEM_PROMPT, TRANSLATION_USER_PROMPT

    llm = create_llm_from_config(llm_config) if (llm_config and llm_config.get("api_base")) else create_llm()

    # Parse source VTT into segments
    segments = _parse_vtt_segments(source_vtt_content)
    logger.info(f"[Translation] {len(segments)} segments, {source_language} → {target_language}")

    system_prompt = TRANSLATION_SYSTEM_PROMPT.format(
        source_language=source_language,
        target_language=target_language,
    )

    # Translate in batches to respect LLM context limits
    translated_segments: list[dict] = []
    for batch_start in range(0, len(segments), _TRANSLATION_BATCH_SIZE):
        batch = segments[batch_start: batch_start + _TRANSLATION_BATCH_SIZE]
        segments_block = _format_segments_for_translation(batch)

        user_prompt = TRANSLATION_USER_PROMPT.format(
            source_language=source_language,
            target_language=target_language,
            segments_block=segments_block,
        )

        response = llm.invoke([("system", system_prompt), ("human", user_prompt)])
        translated_texts = _parse_translation_response(response.content.strip(), len(batch))

        for seg, translated_text in zip(batch, translated_texts):
            translated_segments.append({
                "start": seg["start"],
                "end":   seg["end"],
                "text":  translated_text,
            })

    logger.info(f"[Translation] Done: {len(translated_segments)} segments translated")
    return _build_vtt(translated_segments)


# ─── Whisper helpers ──────────────────────────────────────────────────────────

def _transcribe_video_plain(video_url: str, stt_config: STTConfig | None) -> str:
    """Plain text transcription for the quiz pipeline (no timestamps needed)."""
    api_key = (stt_config.api_key if stt_config else "") or _get_fallback_key()
    model_name = (stt_config.model_name if stt_config else "") or "whisper-1"

    if not api_key:
        raise ValueError(
            "STT API key is not configured. "
            "Set STT_ApiKey in Edunary Admin → System Settings."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")
        logger.info(f"[STT] Extracting audio from: {video_url}")
        _extract_audio_ffmpeg(video_url, audio_path)
        logger.info(f"[STT] Transcribing with model={model_name}")
        transcript = _call_whisper_api_text(audio_path, api_key, model_name)

    logger.info(f"[STT] Transcript length: {len(transcript)} chars")
    return transcript


def _extract_audio_ffmpeg(video_url: str, output_path: str) -> None:
    """Extract full audio as MP3 from a video URL using FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_url,
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "64k",
        output_path,
    ]
    _run_ffmpeg(cmd)


def _extract_audio_chunk(input_path: str, output_path: str, start_sec: float, duration_sec: float) -> None:
    """Extract a time-bounded chunk from a local audio file."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ss", str(start_sec),
        "-t",  str(duration_sec),
        "-c", "copy",
        output_path,
    ]
    _run_ffmpeg(cmd)


def _run_ffmpeg(cmd: list[str]) -> None:
    """Run an FFmpeg command and raise on failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr[-500:]}")
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg is not installed or not in PATH. "
            "Install it to enable video transcription."
        )


def _get_audio_duration(audio_path: str) -> float:
    """Use ffprobe to get the duration of a local audio file in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        # Fallback: assume max duration to trigger chunking conservatively
        return _CHUNK_DURATION_SECS + 1


def _call_whisper_api_text(audio_path: str, api_key: str, model_name: str) -> str:
    """Call Whisper API and return plain text (for quiz pipeline)."""
    with open(audio_path, "rb") as f:
        response = httpx.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            data={"model": model_name, "response_format": "text"},
            files={"file": ("audio.mp3", f, "audio/mpeg")},
            timeout=180.0,
        )
    if response.status_code != 200:
        raise RuntimeError(f"Whisper API error {response.status_code}: {response.text[:300]}")
    return response.text.strip()


def _call_whisper_api_verbose(audio_path: str, api_key: str, model_name: str) -> dict:
    """Call Whisper API with verbose_json to get language + timestamped segments."""
    with open(audio_path, "rb") as f:
        response = httpx.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            data={"model": model_name, "response_format": "verbose_json"},
            files={"file": ("audio.mp3", f, "audio/mpeg")},
            timeout=180.0,
        )
    if response.status_code != 200:
        raise RuntimeError(f"Whisper API error {response.status_code}: {response.text[:300]}")
    return response.json()


# ─── VTT helpers ──────────────────────────────────────────────────────────────

def _build_vtt(segments: list[dict]) -> str:
    """Build a WebVTT string from a list of {start, end, text} segments."""
    lines = ["WEBVTT", ""]
    for i, seg in enumerate(segments, start=1):
        start = _seconds_to_vtt_time(seg["start"])
        end   = _seconds_to_vtt_time(seg["end"])
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def _seconds_to_vtt_time(seconds: float) -> str:
    """Convert float seconds to VTT timestamp HH:MM:SS.mmm."""
    ms = int((seconds % 1) * 1000)
    s  = int(seconds) % 60
    m  = int(seconds // 60) % 60
    h  = int(seconds // 3600)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _parse_vtt_segments(vtt_content: str) -> list[dict]:
    """Parse a VTT string into a list of {start, end, text} dicts."""
    segments = []
    lines = vtt_content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "-->" in line:
            parts = line.split("-->")
            start = _vtt_time_to_seconds(parts[0].strip())
            end   = _vtt_time_to_seconds(parts[1].strip().split()[0])
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1
            text = " ".join(text_lines)
            if text:
                segments.append({"start": start, "end": end, "text": text})
        else:
            i += 1
    return segments


def _vtt_time_to_seconds(vtt_time: str) -> float:
    """Convert VTT timestamp HH:MM:SS.mmm to float seconds."""
    parts = vtt_time.replace(",", ".").split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(parts[0])


# ─── Translation helpers ──────────────────────────────────────────────────────

def _format_segments_for_translation(segments: list[dict]) -> str:
    """Format segments into a numbered block for the LLM translation prompt."""
    blocks = []
    for i, seg in enumerate(segments, start=1):
        blocks.append(f"[{i}]\n{seg['text']}")
    return "\n---\n".join(blocks)


def _parse_translation_response(response_text: str, expected_count: int) -> list[str]:
    """
    Parse LLM translation response back into a list of translated texts.
    Falls back to splitting by separator if numbered format is not found.
    """
    # Try to parse numbered blocks [1], [2], ...
    numbered = re.split(r"\[(\d+)\]", response_text)
    texts = []
    for j in range(1, len(numbered), 2):
        text = numbered[j + 1].strip().replace("---", "").strip() if j + 1 < len(numbered) else ""
        if text:
            texts.append(text)

    if len(texts) == expected_count:
        return texts

    # Fallback: split by separator
    fallback = [t.strip() for t in response_text.split("---") if t.strip()]
    if len(fallback) == expected_count:
        return fallback

    # Last resort: return as many as we have, pad with empty strings
    while len(fallback) < expected_count:
        fallback.append("")
    return fallback[:expected_count]


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _get_fallback_key() -> str:
    """Fallback to .env OPENAI_KEY if no stt_config key provided."""
    from app.core.config import get_settings
    return getattr(get_settings(), "OPENAI_KEY", "")


def _truncate(text: str) -> str:
    """Truncate text to MAX_CONTENT_CHARS with a notice."""
    if len(text) <= MAX_CONTENT_CHARS:
        return text
    logger.warning(f"[Transcription] Content truncated from {len(text)} to {MAX_CONTENT_CHARS} chars")
    return text[:MAX_CONTENT_CHARS] + "\n\n[Content truncated for token limit]"
