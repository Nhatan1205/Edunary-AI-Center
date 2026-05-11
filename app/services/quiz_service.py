"""
Quiz generation service — multi-agent pipeline.

Agents:
  1. Transcription  : Get plain text from lecture_content or video STT.
  2. Retrieval (RAG): Chunk text → embed → temp Qdrant collection → retrieve.
  3. Generator      : LLM drafts questions from retrieved chunks + objectives.
  4. Validator      : Second LLM pass audits and fixes schema/quality issues.
  5. Formatter      : Parse JSON → Pydantic models → return.
"""

from __future__ import annotations

import json
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger, timer
from app.llm.factory import create_llm, create_llm_from_config
from app.embeddings.factory import build_embedding
from app.services.transcription_service import get_lecture_text
from app.prompts.quiz_prompts import (
    GENERATOR_SYSTEM_PROMPT, GENERATOR_USER_PROMPT,
    VALIDATOR_SYSTEM_PROMPT, VALIDATOR_USER_PROMPT,
)
from app.schemas.quiz_schema import (
    GenerateQuizRequest,
    GenerateQuizData,
    GeneratedQuestion,
    GeneratedChoice,
)

logger = get_logger(__name__)

_CHUNK_SIZE    = 800
_CHUNK_OVERLAP = 150
_TOP_K_CHUNKS  = 8     # Number of relevant chunks to retrieve for the LLM


def generate_quiz(request: GenerateQuizRequest) -> GenerateQuizData:
    """
    Orchestrate the multi-agent quiz generation pipeline.

    Returns:
        GenerateQuizData with validated questions.

    Raises:
        ValueError: On content/schema failures.
    """
    with timer("Quiz generation pipeline"):

        # ── Agent 1: Transcription ──────────────────────────────────────────
        logger.info("[Quiz] Agent 1: Transcription")
        lecture_text = get_lecture_text(
            lecture_content=request.lecture_content,
            video_file_url=request.video_file_url,
            stt_config=request.stt_config,
        )
        logger.info(f"[Quiz] Lecture text ready: {len(lecture_text)} chars")

        # ── Agent 2: Retrieval (RAG) ────────────────────────────────────────
        logger.info("[Quiz] Agent 2: RAG — chunking + temp Qdrant collection")
        relevant_context = _retrieve_relevant_chunks(request, lecture_text)

        # ── Agent 3: Generator ──────────────────────────────────────────────
        logger.info("[Quiz] Agent 3: Generator LLM")
        raw_questions_json, content_summary = _run_generator(request, relevant_context)

        # ── Agent 4: Validator ──────────────────────────────────────────────
        logger.info("[Quiz] Agent 4: Validator LLM")
        validated_json = _run_validator(request, relevant_context, raw_questions_json)

        # ── Agent 5: Formatter ──────────────────────────────────────────────
        logger.info("[Quiz] Agent 5: Formatter")
        questions = _parse_and_format(validated_json, request.quiz_config.question_types)

    logger.info(f"[Quiz] Pipeline complete: {len(questions)} questions generated")
    return GenerateQuizData(questions=questions, content_summary=content_summary)


# ─── Agent 2: Retrieval (RAG) ─────────────────────────────────────────────────

def _retrieve_relevant_chunks(request: GenerateQuizRequest, lecture_text: str) -> str:
    """
    Chunk lecture text → embed into a temporary Qdrant collection →
    query with learning objectives → return top-k chunks as context string.
    """
    config = request.quiz_config
    objectives = request.learning_objectives

    # Build search queries from learning objectives + quiz config
    search_queries: list[str] = []
    if objectives.section_level:
        search_queries.append(objectives.section_level)
    if objectives.course_level:
        search_queries.append(objectives.course_level)
    if config.prompt_description:
        search_queries.append(config.prompt_description)
    if not search_queries:
        # Fallback: use lecture title or difficulty
        search_queries.append(
            f"{request.lecture_title} {config.difficulty} level questions"
        )

    # Create a temporary collection name scoped to this request
    temp_collection = f"quiz_tmp_{uuid.uuid4().hex[:12]}"

    try:
        embedding_fn = build_embedding()
        settings = get_settings()
        qdrant_url = settings.QDRANT_URL
        qdrant_api_key = settings.QDRANT_API_KEY or None

        # Chunk the lecture text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=_CHUNK_SIZE,
            chunk_overlap=_CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = splitter.split_text(lecture_text)
        docs = [Document(page_content=c, metadata={"chunk_index": i}) for i, c in enumerate(chunks)]

        logger.info(f"[RAG] {len(chunks)} chunks → temp collection '{temp_collection}'")

        # Use LangChain QdrantVectorStore to index and search
        from langchain_qdrant import QdrantVectorStore
        vector_store = QdrantVectorStore.from_documents(
            docs,
            embedding=embedding_fn,
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=temp_collection,
            force_recreate=True,
        )

        # Multi-query retrieval — merge unique chunks
        seen_content: set[str] = set()
        retrieved: list[str] = []

        for query in search_queries:
            results = vector_store.similarity_search(query, k=_TOP_K_CHUNKS)
            for doc in results:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    retrieved.append(doc.page_content)

        logger.info(f"[RAG] Retrieved {len(retrieved)} unique chunks across {len(search_queries)} queries")

        # Cap total context to avoid token overflow
        context = "\n\n---\n\n".join(retrieved[:_TOP_K_CHUNKS])
        return context

    finally:
        # Always clean up temporary Qdrant collection
        try:
            from qdrant_client import QdrantClient
            _settings = get_settings()
            _cleanup_client = QdrantClient(
                url=_settings.QDRANT_URL,
                api_key=_settings.QDRANT_API_KEY or None,
            )
            _cleanup_client.delete_collection(temp_collection)
            logger.info(f"[RAG] Temp collection '{temp_collection}' deleted")
        except Exception as e:
            logger.warning(f"[RAG] Failed to delete temp collection: {e}")


# ─── Agent 3: Generator ───────────────────────────────────────────────────────

def _run_generator(
    request: GenerateQuizRequest,
    context: str,
) -> tuple[str, str]:
    """
    First LLM pass: draft quiz questions from retrieved context + objectives.

    Returns:
        (raw_questions_json, content_summary)
    """
    config = request.quiz_config
    objectives = request.learning_objectives
    llm_config = request.llm_config

    llm = _get_llm(llm_config)

    user_prompt = GENERATOR_USER_PROMPT.format(
        course_title=request.course_title or "Not specified",
        course_objectives=objectives.course_level or "Not specified",
        section_objectives=objectives.section_level or "Not specified",
        prompt_description=config.prompt_description or "None",
        lecture_title=request.lecture_title or "Not specified",
        lecture_content=context,
        num_questions=config.num_questions,
        question_types=", ".join(config.question_types),
        difficulty=config.difficulty,
    )

    messages = [("system", GENERATOR_SYSTEM_PROMPT), ("human", user_prompt)]
    response = llm.invoke(messages)
    raw = _strip_fences(response.content.strip())
    logger.info(f"[Generator] Response: {len(raw)} chars")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"[Generator] Invalid JSON: {raw[:400]}")
        raise ValueError(f"Generator LLM returned invalid JSON: {e}") from e

    content_summary = data.get("content_summary", "")

    # Generator signalled content/objective mismatch — fail fast
    if content_summary.startswith("CONTENT_MISMATCH"):
        logger.warning(f"[Generator] Content mismatch detected: {content_summary}")
        raise ValueError(
            "The lecture content does not align with the course learning objectives. "
            "Please select a lecture relevant to the course topic, or update the section's "
            "learning objectives to match this lecture."
        )

    questions_raw = json.dumps({"questions": data.get("questions", [])}, ensure_ascii=False)
    return questions_raw, content_summary


# ─── Agent 4: Validator ───────────────────────────────────────────────────────

def _run_validator(
    request: GenerateQuizRequest,
    context: str,
    questions_json: str,
) -> str:
    """
    Second LLM pass: audit and fix schema/pedagogical issues.

    Returns:
        Corrected JSON string.
    """
    objectives = request.learning_objectives
    llm_config = request.llm_config

    llm = _get_llm(llm_config)

    user_prompt = VALIDATOR_USER_PROMPT.format(
        course_objectives=objectives.course_level or "Not specified",
        section_objectives=objectives.section_level or "Not specified",
        lecture_content=context[:3000],   # Limit context for validator pass
        questions_json=questions_json,
    )

    messages = [("system", VALIDATOR_SYSTEM_PROMPT), ("human", user_prompt)]
    response = llm.invoke(messages)
    raw = _strip_fences(response.content.strip())
    logger.info(f"[Validator] Response: {len(raw)} chars")

    # Validate it's still parseable JSON
    try:
        json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[Validator] Returned invalid JSON — falling back to generator output")
        return questions_json

    return raw


# ─── Agent 5: Formatter ───────────────────────────────────────────────────────

def _parse_and_format(
    questions_json: str,
    allowed_types: list[str],
) -> list[GeneratedQuestion]:
    """
    Parse validated JSON → Pydantic models with deterministic post-processing.
    No LLM involved — pure deterministic validation and correction.
    """
    try:
        data = json.loads(questions_json)
        raw_questions = data.get("questions", [])
    except (json.JSONDecodeError, AttributeError) as e:
        raise ValueError(f"Formatter could not parse questions JSON: {e}") from e

    questions: list[GeneratedQuestion] = []
    valid_types = set(allowed_types) if allowed_types else {"SingleChoice", "MultipleChoice", "TrueFalse"}

    for i, q in enumerate(raw_questions):
        q_type = q.get("type", "SingleChoice")

        # Enforce allowed types
        if q_type not in valid_types:
            q_type = next(iter(valid_types))

        choices_raw = q.get("choices", [])
        choices = [
            GeneratedChoice(
                text=str(c.get("text", "")),
                # LLMs sometimes ignore the snake_case instruction → fallback to camelCase
                is_correct=bool(c.get("is_correct") or c.get("isCorrect") or False),
                sort_order=idx,
            )
            for idx, c in enumerate(choices_raw)
        ]

        # Deterministic enforcement of answer counts
        choices = _enforce_answer_counts(choices, q_type)

        questions.append(GeneratedQuestion(
            name=str(q.get("name", f"Question {i + 1}")),
            type=q_type,
            explanation=str(q.get("explanation", "")),
            sort_order=i,
            choices=choices,
        ))

    if not questions:
        raise ValueError("No valid questions were generated. Please try again.")

    return questions


def _enforce_answer_counts(choices: list[GeneratedChoice], q_type: str) -> list[GeneratedChoice]:
    """Deterministically enforce correct answer count rules per question type."""
    correct = [c for c in choices if c.is_correct]
    incorrect = [c for c in choices if not c.is_correct]

    if q_type == "TrueFalse":
        # Must be exactly ["True", "False"] with exactly 1 correct
        true_choice = next((c for c in choices if c.text.lower() == "true"), None)
        false_choice = next((c for c in choices if c.text.lower() == "false"), None)
        has_correct_true = true_choice and true_choice.is_correct
        return [
            GeneratedChoice(text="True", is_correct=bool(has_correct_true), sort_order=0),
            GeneratedChoice(text="False", is_correct=not bool(has_correct_true), sort_order=1),
        ]

    elif q_type == "SingleChoice":
        # Exactly 1 correct
        if len(correct) == 0 and choices:
            choices[0] = GeneratedChoice(text=choices[0].text, is_correct=True, sort_order=0)
        elif len(correct) > 1:
            # Keep only first correct, make rest incorrect
            first_correct = True
            for c in choices:
                if c.is_correct:
                    if first_correct:
                        first_correct = False
                    else:
                        choices[choices.index(c)] = GeneratedChoice(
                            text=c.text, is_correct=False, sort_order=c.sort_order
                        )
        return choices

    elif q_type == "MultipleChoice":
        # At least 1 correct, at most len-1 correct
        if len(correct) == 0 and choices:
            choices[0] = GeneratedChoice(text=choices[0].text, is_correct=True, sort_order=0)
        if len(correct) == len(choices) and choices:
            choices[-1] = GeneratedChoice(text=choices[-1].text, is_correct=False, sort_order=choices[-1].sort_order)
        return choices

    return choices


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_llm(llm_config):
    """Select LLM — backend-provided config, fall back to .env."""
    if llm_config and llm_config.api_base:
        logger.info(f"[LLM] Using backend config: model={llm_config.model_name}")
        return create_llm_from_config(llm_config.model_dump())
    logger.info("[LLM] Using default .env LLM provider")
    return create_llm()


def _strip_fences(raw: str) -> str:
    """Strip markdown code fences if LLM wraps output in ```json ... ```."""
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return raw
