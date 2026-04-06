"""
Course search service.
Handles entity extraction, backend API search, and LLM summarization.
Extracted from intent_handlers.py: handle_get_course_info().
"""

import json
import re

from app.prompts.chat_prompts import ENTITY_SEARCH_PROMPT, COURSE_SUMMARY_PROMPT
from app.llm.factory import create_llm
from app.store.backend_api_client import BackendAPIClient
from app.core.logging import get_logger
from app.core.exceptions import BackendAPIError

logger = get_logger(__name__)


def _clean_json_response(text: str) -> str:
    """
    Clean JSON response from LLM by removing markdown code blocks.

    Handles formats like:
    - ```json\n{...}\n```
    - ```\n{...}\n```
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

    logger.info(f"Extracted topic: '{topic}'")
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
    1. Extract topic from user message
    2. Search courses via backend API
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

        if not topic:
            return {
                "reply": "I am not sure which topic you are looking for. Could you please specify a subject (e.g., Python, Marketing, Web Development)?",
                "messageType": "text",
                "courses": None,
            }

    except json.JSONDecodeError:
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

    # Step 2: Search via backend API
    try:
        client = BackendAPIClient()
        data = client.search_courses(topic)
        items = data.get("items", [])

        if not items:
            return {
                "reply": f"I searched for '{topic}' but could not find any matching courses. Try searching for 'Python', 'Marketing', or 'Design'.",
                "messageType": "text",
                "courses": None,
            }

        # Step 3: Prepare data and summarize
        simplified_courses = []
        for item in items:
            simplified_courses.append({
                "title": item.get("title"),
                "instructor": item.get("instructorName"),
                "price": item.get("price"),
            })

        courses_str = json.dumps(simplified_courses, ensure_ascii=False)
        summary_text = _summarize_courses(message, history, courses_str)

        # Step 4: Build course card (first course only)
        first_item = items[0]
        course_card = {
            "id": first_item.get("id"),
            "title": first_item.get("title", "Untitled Course"),
            "subtitle": first_item.get("subtitle"),
            "imageUrl": first_item.get("imageUrl"),
            "instructorName": first_item.get("instructorName", "Unknown"),
            "courseUrl": client.get_course_url(first_item.get("id")),
        }

        return {
            "reply": summary_text,
            "messageType": "course_cards",
            "courses": [course_card],
        }

    except BackendAPIError:
        return {
            "reply": "Sorry, I cannot connect to the course database right now.",
            "messageType": "text",
            "courses": None,
        }
    except Exception:
        return {
            "reply": "The system is currently undergoing maintenance. Please check back later.",
            "messageType": "text",
            "courses": None,
        }
