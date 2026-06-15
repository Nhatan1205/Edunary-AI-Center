"""
Course Assistant Service — main orchestration layer.
Handles intent dispatch, RAG, LLM generation, and history persistence.
Completely independent from chat_service.py (chatbot).
"""

from app.schemas.curriculum_embedding_schema import CurriculumSearchResult
from typing import Optional, List

from app.llm.factory import create_llm, create_llm_from_config
from app.core.logging import get_logger
from app.guardrails.input_guard import sanitize_input, check_input_length, check_prompt_injection
from app.guardrails.output_guard import check_output_safety, get_fallback_reply
from app.services.curriculum_embedding_service import search_related_lectures, get_lecture_chunks
from app.schemas.model_config import ModelConfig
from app.schemas.course_assistant_schema import CourseAssistantCourseData
from app.store.course_assistant_store import (
    get_or_create_conversation,
    add_message,
    get_recent_messages,
)
from app.prompts.course_assistant_prompts import (
    COURSE_SUMMARY_PROMPT,
    EXPLAIN_LECTURE_PROMPT,
    LEARNING_OBJECTIVES_PROMPT,
    GENERAL_COURSE_QA_PROMPT,
    GENERATE_QUIZ_PROMPT,
    COURSE_ASSISTANT_INTENT_PROMPT,
)

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_HISTORY_LIMIT = 10   # messages loaded for LLM context
_RAG_LIMIT = 5        # lecture chunks to retrieve from Qdrant

VALID_INTENTS = {
    "greeting",
    "course_summary",
    "explain_lecture",
    "learning_objectives",
    "generate_quiz",
    "general_course_qa",
    "unsafe",
    "out_of_scope",
}


# ── Public entry point ────────────────────────────────────────────────────────

def process_assistant_message(
    user_id: str,
    course_id: int,
    course_data: CourseAssistantCourseData,
    content_id: Optional[str],
    content_type: Optional[str],
    media_type: Optional[str],
    content_title: Optional[str],
    message: str,
    llm_config: Optional[ModelConfig] = None,
) -> dict:
    """
    Full pipeline for processing a student's message.

    Steps:
      1. Input guardrails
      2. Get/create conversation in MongoDB
      3. Load recent history for LLM context
      4. Classify intent
      5. Dispatch to intent handler
      6. Output guardrail
      7. Persist both messages to MongoDB
      8. Return response dict
    """
    logger.info(f"[CourseAssistant] START user={user_id} course={course_id} intent pending")

    # 1. Input guardrails
    guarded = _apply_input_guards(message)
    if guarded:
        return guarded

    message = sanitize_input(message)

    # 2. Conversation
    conversation_id = get_or_create_conversation(user_id, course_id, course_data.title)

    # 3. History
    history_docs = get_recent_messages(conversation_id, limit=_HISTORY_LIMIT)
    history_text = _format_history(history_docs)

    # Resolve LLM instance
    llm = _get_llm(llm_config)

    # 4. Intent
    # Normalize content_type: video/article/lecture/empty maps to "lecture" or respective type
    safe_content_type = "lecture"
    if content_type:
        ct_lower = content_type.strip().lower()
        if ct_lower in ("quiz", "assignment"):
            safe_content_type = ct_lower
        elif ct_lower in ("video", "article", "lecture"):
            safe_content_type = "lecture"
        else:
            safe_content_type = ct_lower

    # Use content_label to provide clean contextual information to prompts
    if safe_content_type == "lecture":
        content_label = f"{media_type or 'video'} lecture"
    else:
        content_label = safe_content_type

    safe_content_title = content_title or "Unknown"

    intent, intent_prompt = classify_intent(
        llm=llm,
        message=message,
        course_title=course_data.title,
        content_type=content_label,
        content_title=safe_content_title,
        conversation_history=history_text,
    )

    # 5. Dispatch
    result = _dispatch(
        llm=llm,
        intent=intent,
        message=message,
        course_id=course_id,
        course_data=course_data,
        content_id=content_id,
        content_type=safe_content_type,
        content_label=content_label,
        content_title=safe_content_title,
        history_text=history_text,
    )

    # 6. Output guardrail
    is_safe, _ = check_output_safety(result["reply"])
    if not is_safe:
        result["reply"] = get_fallback_reply()
        result["sources"] = None

    # Write debug log to file
    # _save_debug_log(
    #     user_id=user_id,
    #     course_id=course_id,
    #     course_data=course_data,
    #     content_id=content_id,
    #     content_type_raw=content_type,
    #     content_type_normalized=safe_content_type,
    #     media_type=media_type,
    #     content_label=content_label,
    #     content_title=content_title,
    #     message=message,
    #     intent=intent,
    #     intent_prompt=intent_prompt,
    #     result=result,
    # )

    # 7. Persist
    add_message(
        conversation_id=conversation_id,
        role="user",
        content=message,
        content_id=content_id,
        content_type=safe_content_type,
    )
    add_message(
        conversation_id=conversation_id,
        role="assistant",
        content=result["reply"],
        sources=result.get("sources"),
    )

    logger.info(f"[CourseAssistant] DONE intent={intent} message_type={result['message_type']}")
    return result


# ── Dispatch ──────────────────────────────────────────────────────────────────

def _dispatch(
    llm,
    intent: str,
    message: str,
    course_id: int,
    course_data: CourseAssistantCourseData,
    content_id: Optional[str],
    content_type: str,
    content_label: str,
    content_title: str,
    history_text: str,
) -> dict:
    """Route to the appropriate handler based on classified intent."""

    if intent == "greeting":
        return _handle_greeting(course_data.title)

    if intent == "unsafe":
        return _handle_unsafe()

    if intent == "out_of_scope":
        return _handle_out_of_scope(course_data.title)

    if intent == "generate_quiz":
        return _handle_generate_quiz(
            llm=llm,
            course_id=course_id,
            course_title=course_data.title,
            content_id=content_id,
            content_type=content_type,
            content_title=content_title,
        )

    # All remaining intents use RAG + LLM
    return _handle_rag_intent(
        llm=llm,
        intent=intent,
        message=message,
        course_id=course_id,
        course_data=course_data,
        content_id=content_id,
        content_type=content_type,
        content_label=content_label,
        content_title=content_title,
        history_text=history_text,
    )


# ── Intent Handlers ───────────────────────────────────────────────────────────

def _handle_greeting(course_title: str) -> dict:
    return {
        "reply": f"Hi there! 👋 I'm your AI learning assistant for **{course_title}**. Feel free to ask me anything about this course — I'm here to help you learn! 😊",
        "message_type": "text",
        "sources": None,
        "rendered_prompt": "(No LLM call for greeting intent)",
        "raw_results": [],
    }


def _handle_unsafe() -> dict:
    return {
        "reply": "I can't respond to that. I'm here to help you learn — feel free to ask anything about the course! 😊",
        "message_type": "text",
        "sources": None,
        "rendered_prompt": "(No LLM call for unsafe intent)",
        "raw_results": [],
    }


def _handle_out_of_scope(course_title: str) -> dict:
    return {
        "reply": f"I can only help with questions about **{course_title}**. Try asking about a lecture topic, concept, or your progress in the course!",
        "message_type": "text",
        "sources": None,
        "rendered_prompt": "(No LLM call for out-of-scope intent)",
        "raw_results": [],
    }


def _handle_rag_intent(
    llm,
    intent: str,
    message: str,
    course_id: int,
    course_data: CourseAssistantCourseData,
    content_id: Optional[str],
    content_type: str,
    content_label: str,
    content_title: str,
    history_text: str,
) -> dict:
    """
    Handles: course_summary, explain_lecture, learning_objectives, general_course_qa.
    Runs RAG against edunary_curriculum, then calls LLM with appropriate prompt.
    """
    # RAG
    rag_context, sources, raw_results = _retrieve_context(
        message=message,
        course_id=course_id,
        content_id=content_id,
        intent=intent,
    )

    # Fallback when course content not embedded yet
    if not rag_context:
        if content_type in ("quiz", "assignment"):
            return {
                "reply": "I don't have detailed content for this quiz/assignment yet. Try asking about a specific lecture topic instead! 📖",
                "message_type": "text",
                "sources": None,
                "rendered_prompt": "(No LLM call - quiz/assignment context not found)",
                "raw_results": [],
            }
        return {
            "reply": f"I couldn't find relevant content in the course materials for that question. Try rephrasing, or ask about a specific topic from **{course_data.title}**.",
            "message_type": "text",
            "sources": None,
            "rendered_prompt": "(No LLM call - context not found)",
            "raw_results": [],
        }

    # Select prompt by intent
    prompt_map = {
        "course_summary": COURSE_SUMMARY_PROMPT,
        "explain_lecture": EXPLAIN_LECTURE_PROMPT,
        "learning_objectives": LEARNING_OBJECTIVES_PROMPT,
    }
    prompt = prompt_map.get(intent, GENERAL_COURSE_QA_PROMPT)

    variables = {
        "course_title": course_data.title,
        "course_subtitle": course_data.subtitle or "",
        "course_description": course_data.description or "",
        "course_learning_objectives": course_data.learning_objectives or "",
        "course_requirements": course_data.requirements or "",
        "course_target_audience": course_data.target_audience or "",
        "content_type": content_label,
        "content_title": content_title,
        "rag_context": rag_context,
        "conversation_history": history_text,
        "message": message,
    }
    rendered_prompt = prompt.format(**variables)
    reply = _call_llm(llm, prompt, variables)

    return {
        "reply": reply,
        "message_type": "text",
        "sources": sources,
        "rendered_prompt": rendered_prompt,
        "raw_results": raw_results,
    }


def _handle_generate_quiz(
    llm,
    course_id: int,
    course_title: str,
    content_id: Optional[str],
    content_type: str,
    content_title: str,
) -> dict:
    """
    Generates 2 inline practice questions from the current lecture's RAG context.
    Lightweight — no full quiz pipeline, no temp Qdrant collection.
    """
    if content_type in ("quiz", "assignment"):
        return {
            "reply": "I don't have detailed content for this quiz/assignment yet. Try navigating to a lecture first, then ask me to generate practice questions! 📖",
            "message_type": "text",
            "sources": None,
            "rendered_prompt": "(No LLM call - generating quiz not applicable for quiz/assignment)",
            "raw_results": [],
        }

    rag_context, sources, raw_results = _retrieve_context(
        message=content_title or "lecture content",
        course_id=course_id,
        content_id=content_id,
        intent="generate_quiz",
    )

    if not rag_context:
        return {
            "reply": "I couldn't find lecture content to generate questions from. Make sure the course curriculum has been embedded, or try a different lecture.",
            "message_type": "text",
            "sources": None,
            "rendered_prompt": "(No LLM call - context not found for quiz generation)",
            "raw_results": [],
        }

    variables = {
        "course_title": course_title,
        "content_title": content_title or "this lecture",
        "rag_context": rag_context,
    }
    rendered_prompt = GENERATE_QUIZ_PROMPT.format(**variables)
    reply = _call_llm(llm, GENERATE_QUIZ_PROMPT, variables)

    return {
        "reply": reply,
        "message_type": "quiz",
        "sources": sources,
        "rendered_prompt": rendered_prompt,
        "raw_results": raw_results,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _apply_input_guards(message: str) -> Optional[dict]:
    """Run input guardrails. Returns an error dict if blocked, else None."""
    if check_input_length(message):
        return {
            "reply": "⚠️ Your message is too long. Please keep it under 2000 characters.",
            "message_type": "text",
            "sources": None,
        }
    if check_prompt_injection(message):
        return {
            "reply": "I can only help with questions about this course. How can I assist you? 😊",
            "message_type": "text",
            "sources": None,
        }
    return None


def _format_history(history_docs: List[dict]) -> str:
    """Convert message docs to a formatted string for LLM context."""
    if not history_docs:
        return ""
    lines = []
    for doc in history_docs:
        role = "Student" if doc.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {doc.get('content', '')}")
    return "\n".join(lines)


def _retrieve_context(
    message: str,
    course_id: int,
    content_id: Optional[str],
    intent: Optional[str] = None,
) -> tuple[str, List[str], List[CurriculumSearchResult]]:
    """
    Search edunary_curriculum for relevant lecture chunks filtered by course_id.
    Includes direct matching on content_id for lecture-specific queries.
    Returns (context_text, source_titles, raw_results).
    """
    try:
        results = []
        is_item_specific = False

        if intent in ("explain_lecture", "generate_quiz"):
            is_item_specific = True
        else:
            msg_lower = message.lower()
            if "this lecture" in msg_lower or "this video" in msg_lower or "this article" in msg_lower or "here" in msg_lower:
                is_item_specific = True

        # 1. Direct match for current lecture if content_id is provided and query is specific
        if content_id and is_item_specific:
            direct_results = get_lecture_chunks(course_id=course_id, item_id=content_id)
            if direct_results:
                results.extend(direct_results)
                logger.info(f"[CourseAssistant] Retrieved {len(direct_results)} direct chunks for item {content_id}")
                needed = 0  # ONLY use these direct chunks, do not mix with other lectures!
            else:
                needed = _RAG_LIMIT
        else:
            needed = _RAG_LIMIT

        # 2. Semantic search to fill the remaining slots
        if needed > 0:
            semantic_results = search_related_lectures(
                query_text=message,
                course_id=course_id,
                limit=needed,
            )
            existing_ids = {r.item_id for r in results}
            for r in semantic_results:
                if r.item_id not in existing_ids:
                    results.append(r)

        if not results:
            return "", [], []

        chunks = [r.content_text for r in results if r.content_text]
        context = "\n\n---\n\n".join(chunks)

        sources = list({
            f"{r.section_title} — {r.item_title}"
            for r in results
            if r.section_title or r.item_title
        })

        logger.info(f"[CourseAssistant] RAG complete: {len(results)} chunks, {len(sources)} sources")
        return context, sources, results

    except Exception as e:
        logger.error(f"[CourseAssistant] RAG failed: {e}")
        return "", [], []


def _call_llm(llm, prompt, variables: dict) -> str:
    """Invoke the LLM with given prompt template and variables."""
    result = (prompt | llm).invoke(variables)
    text = result.content if hasattr(result, "content") else str(result)
    return text.strip()


def _get_llm(llm_config: Optional[ModelConfig]):
    """Select LLM — backend-provided config, fall back to .env."""
    if llm_config and llm_config.api_base:
        logger.info(f"[LLM] Using backend config: model={llm_config.model_name}")
        return create_llm_from_config(llm_config.model_dump())
    logger.info("[LLM] Using default .env LLM provider")
    return create_llm()


def classify_intent(
    llm,
    message: str,
    course_title: str,
    content_type: str,
    content_title: str,
    conversation_history: str,
) -> tuple[str, str]:
    variables = {
        "course_title": course_title or "this course",
        "content_type": content_type or "lecture",
        "content_title": content_title or "unknown",
        "conversation_history": conversation_history or "(no previous messages)",
        "message": message,
    }
    rendered_prompt = COURSE_ASSISTANT_INTENT_PROMPT.format(**variables)
    result = (COURSE_ASSISTANT_INTENT_PROMPT | llm).invoke(variables)

    raw = result.content if hasattr(result, "content") else str(result)
    intent = raw.strip().lower()

    if intent not in VALID_INTENTS:
        logger.warning(f"[CourseAssistantIntent] Unknown intent '{intent}' — defaulting to general_course_qa")
        intent = "general_course_qa"

    logger.info(f"[CourseAssistantIntent] intent='{intent}'")
    return intent, rendered_prompt


def _save_debug_log(
    user_id: str,
    course_id: int,
    course_data: CourseAssistantCourseData,
    content_id: Optional[str],
    content_type_raw: Optional[str],
    content_type_normalized: str,
    media_type: Optional[str],
    content_label: str,
    content_title: Optional[str],
    message: str,
    intent: str,
    intent_prompt: str,
    result: dict,
) -> None:
    """Saves RAG results, prompts, and output to a local JSON debug log file."""
    try:
        import os
        import json
        from datetime import datetime

        raw_results = result.get("raw_results", [])
        retrieved_chunks = [
            {
                "section_id": r.section_id,
                "section_title": r.section_title,
                "item_id": r.item_id,
                "item_title": r.item_title,
                "item_type": r.item_type,
                "content_text": r.content_text,
                "score": r.score
            }
            for r in raw_results
        ]

        debug_info = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "course_id": course_id,
            "course_title": course_data.title,
            "course_subtitle": course_data.subtitle,
            "course_description": course_data.description,
            "course_learning_objectives": course_data.learning_objectives,
            "course_requirements": course_data.requirements,
            "course_target_audience": course_data.target_audience,
            "content_id": content_id,
            "content_type_raw": content_type_raw,
            "content_type_normalized": content_type_normalized,
            "media_type": media_type,
            "content_label": content_label,
            "content_title": content_title,
            "message": message,
            "classified_intent": intent,
            "intent_classifier_prompt": intent_prompt,
            "retrieved_chunks": retrieved_chunks,
            "generation_prompt": result.get("rendered_prompt"),
            "reply": result.get("reply")
        }

        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(app_dir, "debug_logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"course_assistant_debug_{timestamp_str}.json"
        log_filepath = os.path.join(log_dir, log_filename)

        with open(log_filepath, "w", encoding="utf-8") as f:
            json.dump(debug_info, f, ensure_ascii=False, indent=2)

        logger.info(f"[CourseAssistantDebug] Saved debug log to: {log_filepath}")

    except Exception as ex:
        logger.error(f"[CourseAssistantDebug] Failed to save debug log: {ex}")
