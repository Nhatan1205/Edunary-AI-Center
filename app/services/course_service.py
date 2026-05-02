"""
Course search service.
Handles entity extraction from user message, vector similarity search via Qdrant,
and LLM summarization of results.

Previously used BackendAPIClient.search_courses() (HTTP call to .NET backend).
Now uses the edunary_courses Qdrant collection for semantic similarity search.
BackendAPIClient is kept for potential future use.
"""

import json
import re

from app.prompts.chat_prompts import ENTITY_SEARCH_PROMPT, COURSE_SUMMARY_PROMPT
from app.llm.factory import create_llm
from app.core.logging import get_logger
from app.store.backend_api_client import BackendAPIClient  # kept for future use
from app.services.course_embedding_service import search_courses

logger = get_logger(__name__)

# Maximum number of courses returned from vector search
_SEARCH_LIMIT = 5


def _clean_json_response(text: str) -> str:
    """
    Clean JSON response from LLM by removing markdown code blocks.

    Handles formats like:
    - ```json\\n{...}\\n```
    - ```\\n{...}\\n```
    - Just {...}
    """
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()


def _extract_topic(message: str, history: str) -> str | None:
    """
    Extract the search topic from the user's message using LLM.

    Returns:
        str or None: The extracted topic, or None if not found.
    """
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


def _summarize_courses(message: str, history: str, courses_json: str) -> str:
    """Generate a natural language summary of found courses using LLM."""
    llm = create_llm()

    summary_res = (COURSE_SUMMARY_PROMPT | llm).invoke({
        "chat_history": history,
        "input": message,
        "courses_json": courses_json,
    })

    return (summary_res.content if hasattr(summary_res, "content") else summary_res).strip()


def handle_course_search(message: str, history: str) -> dict:
    """
    Full course search pipeline:
    1. Extract topic from user message (LLM)
    2. Search courses via Qdrant vector similarity (edunary_courses collection)
    3. Summarize results with LLM
    4. Return response dict with course cards

    Args:
        message: User's message.
        history: Formatted conversation history.

    Returns:
        dict with keys: reply, messageType, courses
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
            }

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error extracting topic: {e}")
        return {
            "reply": "Sorry, the system is busy. Please try again later.",
            "messageType": "text",
            "courses": None,
        }
    except Exception:
        return {
            "reply": "An error occurred while processing your request.",
            "messageType": "text",
            "courses": None,
        }

    # Step 2: Semantic search via Qdrant
    try:
        results = search_courses(query=topic, limit=_SEARCH_LIMIT)

        if not results:
            return {
                "reply": f"I searched for '{topic}' but could not find any matching courses. Try searching for 'Python', 'Marketing', or 'Design'.",
                "messageType": "text",
                "courses": None,
            }

        # Step 3: Prepare data for LLM summarization
        simplified_courses = [
            {
                "title": r.title,
                "instructor": r.instructor_name,
                "price": r.price,
            }
            for r in results
        ]
        courses_str = json.dumps(simplified_courses, ensure_ascii=False)
        summary_text = _summarize_courses(message, history, courses_str)

        # Step 4: Build course card from the top result
        top = results[0]
        backend_client = BackendAPIClient()
        course_card = {
            "id": top.course_id,
            "title": top.title,
            "subtitle": top.subtitle,
            "imageUrl": top.image_url,
            "instructorName": top.instructor_name,
            "courseUrl": backend_client.get_course_url(top.course_id),
        }

        return {
            "reply": summary_text,
            "messageType": "course_cards",
            "courses": [course_card],
        }

    except Exception as e:
        logger.error(f"Course search via Qdrant failed: {e}")
        return {
            "reply": "The system is currently undergoing maintenance. Please check back later.",
            "messageType": "text",
            "courses": None,
        }
