"""
Course search service.
Handles entity extraction from user message, vector similarity search via Qdrant,
and LLM summarization of results.

Supports both anonymous (generic) and personalized (authenticated user) flows.
"""

import json
import re
from typing import TYPE_CHECKING

from app.prompts.chat_prompts import (
    ENTITY_SEARCH_PROMPT,
    COURSE_SUMMARY_PROMPT,
    PERSONALIZED_COURSE_SUMMARY_PROMPT,
    PERSONALIZED_RECOMMENDATION_PROMPT,
)
from app.llm.factory import create_llm
from app.core.logging import get_logger
from app.store.backend_api_client import BackendAPIClient  # kept for future use
from app.services.course_embedding_service import search_courses

if TYPE_CHECKING:
    from app.services.user_profile_service import UserProfileContext

logger = get_logger(__name__)

# Maximum number of courses returned from vector search
_SEARCH_LIMIT = 5


def _clean_json_response(text: str) -> str:
    """Clean JSON response from LLM by removing markdown code blocks."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()


def _extract_topic(message: str, history: str) -> str | None:
    """Extract the search topic from the user's message using LLM."""
    llm = create_llm()

    raw_result = (ENTITY_SEARCH_PROMPT | llm).invoke({
        "chat_history": history,
        "input": message,
    })

    result_content = raw_result.content if hasattr(raw_result, "content") else raw_result
    clean_json = _clean_json_response(result_content)

    entity_data = json.loads(clean_json)
    topic = entity_data.get("topic", "")

    return topic if topic else None


def _summarize_courses(
    message: str,
    courses_json: str,
    user_context: "UserProfileContext | None" = None,
) -> str:
    """Generate a natural language summary of found courses using LLM.
    Uses personalized prompt when user context is available.
    """
    llm = create_llm()

    if user_context:
        summary_res = (PERSONALIZED_COURSE_SUMMARY_PROMPT | llm).invoke({
            "input": message,
            "user_context": user_context.to_prompt_text(),
            "courses_json": courses_json,
        })
    else:
        summary_res = (COURSE_SUMMARY_PROMPT | llm).invoke({
            "input": message,
            "courses_json": courses_json,
        })

    return (summary_res.content if hasattr(summary_res, "content") else summary_res).strip()


def handle_course_search(
    message: str,
    history: str,
    user_context: "UserProfileContext | None" = None,
) -> dict:
    """
    Full course search pipeline:
    1. Extract topic from user message (LLM)
    2. Search courses via Qdrant vector similarity
    3. Summarize results with LLM (personalized if user_context provided)
    4. Return response dict with course cards
    """
    # Step 1: Extract topic
    try:
        topic = _extract_topic(message, history)
        logger.info(f"Extracted topic: '{topic}'")

        if not topic:
            logger.warning("Topic is empty — returning early, Qdrant not queried")
            return {
                "reply": "I am not sure which topic you are looking for. Could you please specify a subject (e.g., Python, Marketing, Web Development)?",
                "messageType": "text",
                "courses": None,
                "is_personalized": user_context is not None,
            }

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error extracting topic: {e}")
        return {
            "reply": "Sorry, the system is busy. Please try again later.",
            "messageType": "text",
            "courses": None,
            "is_personalized": False,
        }
    except Exception:
        return {
            "reply": "An error occurred while processing your request.",
            "messageType": "text",
            "courses": None,
            "is_personalized": False,
        }

    # Step 2: Semantic search via Qdrant
    try:
        results = search_courses(query=topic, limit=_SEARCH_LIMIT)

        if not results:
            return {
                "reply": f"I searched for '{topic}' but could not find any matching courses. Try searching for 'Python', 'Marketing', or 'Design'.",
                "messageType": "text",
                "courses": None,
                "is_personalized": user_context is not None,
            }

        # Step 3: Prepare simplified data for LLM summarization
        simplified_courses = [
            {
                "title": r.title,
                "instructor": r.instructor_name,
                "price": r.price,
                "level": r.level,
                "category": r.category_name,
            }
            for r in results
        ]
        courses_str = json.dumps(simplified_courses, ensure_ascii=False)
        summary_text = _summarize_courses(message, courses_str, user_context)

        # Step 4: Build course cards (top result as card)
        backend_client = BackendAPIClient()
        course_cards = [
            {
                "id": r.course_id,
                "title": r.title,
                "subtitle": r.subtitle,
                "imageUrl": r.image_url,
                "instructorName": r.instructor_name,
                "courseUrl": backend_client.get_course_url(r.course_id),
            }
            for r in results[:3]  # Show top 3 cards
        ]

        return {
            "reply": summary_text,
            "messageType": "course_cards",
            "courses": course_cards,
            "is_personalized": user_context is not None,
        }

    except Exception as e:
        logger.error(f"Course search via Qdrant failed: {e}")
        return {
            "reply": "The system is currently undergoing maintenance. Please check back later.",
            "messageType": "text",
            "courses": None,
            "is_personalized": False,
        }


def handle_personalized_recommendation(
    user_context: "UserProfileContext",
    history: str,
) -> dict:
    """
    Handle the 'personalized_recommendation' intent.
    Uses the user's preferred_topics + goal to drive a semantic search,
    then summarizes results in a personalized way.

    Only called when user_context is not None.
    """
    # Build a rich query from user profile data
    query_parts: list[str] = []
    if user_context.preferred_topics:
        query_parts.extend(user_context.preferred_topics)
    if user_context.preferred_categories:
        query_parts.extend(user_context.preferred_categories)
    if user_context.goal:
        query_parts.append(user_context.goal)

    query = ", ".join(query_parts) if query_parts else "popular courses"
    logger.info(f"Personalized recommendation query: '{query}' for user '{user_context.display_name}'")

    try:
        results = search_courses(query=query, limit=_SEARCH_LIMIT)

        if not results:
            return {
                "reply": f"Hi {user_context.display_name}! I couldn't find courses matching your current interests right now. Try asking about a specific topic like Python or Marketing.",
                "messageType": "text",
                "courses": None,
                "is_personalized": True,
            }

        simplified_courses = [
            {
                "title": r.title,
                "instructor": r.instructor_name,
                "price": r.price,
                "level": r.level,
                "category": r.category_name,
            }
            for r in results
        ]
        courses_str = json.dumps(simplified_courses, ensure_ascii=False)

        llm = create_llm()
        summary_res = (PERSONALIZED_RECOMMENDATION_PROMPT | llm).invoke({
            "user_context": user_context.to_prompt_text(),
            "courses_json": courses_str,
        })
        summary_text = (summary_res.content if hasattr(summary_res, "content") else summary_res).strip()

        backend_client = BackendAPIClient()
        course_cards = [
            {
                "id": r.course_id,
                "title": r.title,
                "subtitle": r.subtitle,
                "imageUrl": r.image_url,
                "instructorName": r.instructor_name,
                "courseUrl": backend_client.get_course_url(r.course_id),
            }
            for r in results[:3]
        ]

        return {
            "reply": summary_text,
            "messageType": "course_cards",
            "courses": course_cards,
            "is_personalized": True,
        }

    except Exception as e:
        logger.error(f"Personalized recommendation failed: {e}")
        return {
            "reply": "An error occurred while fetching recommendations. Please try again.",
            "messageType": "text",
            "courses": None,
            "is_personalized": True,
        }
