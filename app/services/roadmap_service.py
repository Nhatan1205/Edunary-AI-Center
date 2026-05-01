import json

from app.prompts.roadmap_prompts import SYSTEM_PROMPT, USER_PROMPT
from app.llm.factory import create_llm, create_llm_from_config
from app.core.logging import get_logger, timer
from app.schemas.roadmap_schema import (
    GeneratePersonalizedRoadmapRequest,
    PersonalizedRoadmapData,
    RoadmapNode,
    RoadmapEdge,
)

logger = get_logger(__name__)

def generate_roadmap(
    request: GeneratePersonalizedRoadmapRequest,
) -> PersonalizedRoadmapData:
    profile = request.student_profile
    roadmap = request.roadmap_data
    catalog = request.courses_catalog
    llm_config = request.llm_config

    with timer("Personalized roadmap LLM generation"):
        # Choose LLM — backend-provided config, fall back to .env
        if llm_config and llm_config.api_base:
            llm = create_llm_from_config(llm_config.model_dump())
            logger.info(
                f"Using backend llm_config: model={llm_config.model_name}, "
                f"base={llm_config.api_base}"
            )
        else:
            llm = create_llm()
            logger.info("Using default .env LLM provider for personalized roadmap")

        catalog_list = [
            {
                "id": c.id,
                "title": c.title,
                "level": c.level,
                "topics": c.topics,
                "category": c.category,
                "ratings": c.ratings,
                "total_students": c.total_students,
                "learning_objectives": c.learning_objectives,
            }
            for c in catalog
        ]

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
            f"Personalized roadmap: goal='{profile.goal}', "
            f"level='{profile.skill_level}', topic='{roadmap.roadmap_topic}', "
            f"catalog_size={len(catalog)}"
        )

        response = llm.invoke(messages)
        raw = response.content.strip()

        # Strip markdown code fences if LLM wraps output in ```json ... ```
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"LLM returned invalid JSON: {raw[:300]}")
            raise ValueError(f"LLM returned invalid JSON: {e}") from e

    try:
        nodes = [RoadmapNode(**n) for n in data.get("nodes", [])]
        edges = [RoadmapEdge(**e) for e in data.get("edges", [])]
        summary = data.get("summary", "")
    except Exception as e:
        logger.error(f"LLM JSON structure mismatch: {e}")
        raise ValueError(f"LLM response did not match expected schema: {e}") from e

    logger.info(
        f"Personalized roadmap generated: {len(nodes)} nodes, {len(edges)} edges"
    )

    return PersonalizedRoadmapData(nodes=nodes, edges=edges, summary=summary)
