import json

from app.prompts.roadmap_prompts import SYSTEM_PROMPT, USER_PROMPT
from app.llm.factory import create_llm, create_llm_from_config
from app.core.logging import get_logger, timer
from app.schemas.roadmap_schema import (
    GeneratePersonalizedRoadmapRequest,
    PersonalizedRoadmapData,
    RoadmapNode,
    RoadmapEdge,
    StudentProfile,
    RoadmapData,
)
from app.schemas.course_embedding_schema import CourseSearchResult
from app.services.course_embedding_service import search_courses

logger = get_logger(__name__)


def _search_courses_for_roadmap(roadmap: RoadmapData, profile: StudentProfile) -> list[CourseSearchResult]:
    """
    Multi-query semantic search leveraging ALL personalization data.
    Queries:
        Q1 (limit=10): description + topic — PRIMARY session intent
        Q2 (limit=5):  goal from profile — long-term career direction
        Q3 (limit=5):  preferred_topics + preferred_categories — student interests
    """
    seen_ids: dict[int, CourseSearchResult] = {}  # course_id → best-scored result

    def _merge(results: list[CourseSearchResult]) -> None:
        for r in results:
            if r.course_id not in seen_ids or r.score > seen_ids[r.course_id].score:
                seen_ids[r.course_id] = r

    # Q1: Primary session intent — description + topic (most important)
    primary_query = roadmap.description
    if roadmap.roadmap_topic:
        primary_query += f" {roadmap.roadmap_topic}"
    q1_results = search_courses(query=primary_query, limit=10)
    _merge(q1_results)
    logger.info(f"[Search] Q1 (intent)    : {len(q1_results):2d} courses")

    # Q2: Long-term career goal — may surface courses Q1 misses
    if profile.goal:
        q2_results = search_courses(query=profile.goal, limit=5)
        _merge(q2_results)
        logger.info(f"[Search] Q2 (goal)      : {len(q2_results):2d} courses")
    else:
        logger.info("[Search] Q2 (goal)      : skipped (no goal set)")

    # Q3: Preferred topics + categories — student's known interests
    interest_parts: list[str] = []
    if profile.preferred_topics:
        interest_parts.extend(profile.preferred_topics)
    if profile.preferred_categories:
        interest_parts.extend(profile.preferred_categories)
    if interest_parts:
        interest_query = ", ".join(interest_parts)
        q3_results = search_courses(query=interest_query, limit=5)
        _merge(q3_results)
        logger.info(f"[Search] Q3 (interests) : {len(q3_results):2d} courses")
    else:
        logger.info("[Search] Q3 (interests) : skipped (no preferred topics/categories)")

    # Filter out already-enrolled courses
    enrolled_set = set(profile.enrolled_course_ids)
    filtered = [r for r in seen_ids.values() if r.course_id not in enrolled_set]

    # Sort by relevance score descending, cap at 15
    filtered.sort(key=lambda r: r.score, reverse=True)
    top = filtered[:15]

    logger.info(
        f"[Search] Result         : {len(seen_ids)} unique → {len(top)} passed to LLM "
        f"(excluded {len(enrolled_set)} enrolled)"
    )
    return top


def generate_roadmap(
    request: GeneratePersonalizedRoadmapRequest,
) -> PersonalizedRoadmapData:
    profile = request.student_profile
    roadmap = request.roadmap_data
    llm_config = request.llm_config

    with timer("Personalized roadmap — Qdrant search + LLM generation"):
        # ── Step 1: Multi-query semantic search ──
        search_results = _search_courses_for_roadmap(roadmap, profile)

        if not search_results:
            raise ValueError(
                "No courses found in the Qdrant collection. "
                "Please ensure courses are embedded before generating a roadmap."
            )

        # ── Step 2: Build catalog list for LLM prompt ──
        catalog_list = [
            {
                "id": r.course_id,
                "title": r.title,
                "level": r.level or "Unknown",
                "category": r.category_name or "Unknown",
                "score": round(r.score, 4),  # relevance score — LLM context signal
            }
            for r in search_results
        ]

        # ── Step 3: Choose LLM — backend-provided config, fall back to .env ──
        if llm_config and llm_config.api_base:
            llm = create_llm_from_config(llm_config.model_dump())
            logger.info(
                f"Using backend llm_config: model={llm_config.model_name}, "
                f"base={llm_config.api_base}"
            )
        else:
            llm = create_llm()
            logger.info("Using default .env LLM provider for personalized roadmap")

        # ── Step 4: Build and invoke LLM prompt ──
        user_prompt = USER_PROMPT.format(
            skill_level=profile.skill_level,
            goal=profile.goal or "Not specified",
            preferred_topics=", ".join(profile.preferred_topics) if profile.preferred_topics else "None",
            preferred_categories=", ".join(profile.preferred_categories) if profile.preferred_categories else "None",
            weekly_hours=profile.weekly_hours if profile.weekly_hours is not None else "Not specified",
            enrolled_course_ids=profile.enrolled_course_ids or "[]",
            description=roadmap.description,
            roadmap_topic=roadmap.roadmap_topic,
            courses_catalog_json=json.dumps(catalog_list, ensure_ascii=False, indent=2),
        )

        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", user_prompt),
        ]

        logger.info(
            f"[LLM] Calling with {len(catalog_list)} courses "
            f"| level={profile.skill_level} | topic='{roadmap.roadmap_topic}'"
        )

        response = llm.invoke(messages)
        raw = response.content.strip()

        # Strip markdown code fences if LLM wraps output in ```json ... ```
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        logger.info(f"[LLM] Response received ({len(raw)} chars): {raw[:120]}...")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"[LLM] Invalid JSON: {raw[:300]}")
            raise ValueError(f"LLM returned invalid JSON: {e}") from e

    # ── Step 5: Parse and return structured response ──
    try:
        nodes = [RoadmapNode(**n) for n in data.get("nodes", [])]
        edges = [RoadmapEdge(**e) for e in data.get("edges", [])]
        # summary from LLM becomes the human-readable description of the roadmap.
        # Edunary backend stores this in Roadmap.Description (replacing the raw user prompt).
        description = data.get("summary", "") or roadmap.description
    except Exception as e:
        logger.error(f"LLM JSON structure mismatch: {e}")
        raise ValueError(f"LLM response did not match expected schema: {e}") from e

    searched_course_ids = [r.course_id for r in search_results]

    logger.info(
        f"[Done] {len(nodes)} nodes, {len(edges)} edges "
        f"| searched_ids={searched_course_ids}"
    )

    return PersonalizedRoadmapData(
        nodes=nodes,
        edges=edges,
        summary=description,
        searched_course_ids=searched_course_ids,
    )
