"""
Roadmap generation service.

Calls the LLM with a structured prompt to generate a learning roadmap
in the same graph format used by Edunary's Roadmap entity (GraphData).

The output is a JSON object with `nodes` and `edges` — nodes contain
AI-generated topic labels (not real CourseIds from the DB yet).
The .NET backend receives this and can let the user map nodes to real courses.
"""

import json

from app.llm.factory import create_llm, create_llm_from_config
from app.core.logging import get_logger, timer

logger = get_logger(__name__)

# fmt: off
ROADMAP_SYSTEM_PROMPT = (
    "You are an expert curriculum designer. "
    "Given a learning topic and level, you generate a structured, sequential learning roadmap. "
    "You respond ONLY with valid JSON — no markdown, no explanation, no code fences."
)

ROADMAP_USER_PROMPT = """\
Generate a learning roadmap for the following:

Topic : {topic}
Level : {level}

Return a JSON object with this exact structure:
{{
  "nodes": [
    {{
      "clientNodeId": "node_1",
      "label": "Topic or course title",
      "description": "One sentence describing what the learner will gain",
      "positionX": 250,
      "positionY": 0,
      "sortOrder": 1
    }}
  ],
  "edges": [
    {{
      "sourceNodeId": "node_1",
      "targetNodeId": "node_2"
    }}
  ]
}}

Rules:
- Generate between 5 and 10 nodes.
- positionX is always 250 (center-aligned single column).
- positionY starts at 0 and increases by 120 for each node (0, 120, 240, ...).
- sortOrder starts at 1 and increments by 1.
- Edges represent prerequisites: learn sourceNodeId before targetNodeId.
- Each node label must be a specific, actionable learning topic.
- Respond with JSON only.
"""
# fmt: on


def generate_roadmap(topic: str, level: str, model_config: dict | None = None) -> dict:
    """
    Generate an AI roadmap graph for the given topic and level.

    Args:
        topic       : Learning topic (e.g. "Python", "Web Development")
        level       : Difficulty level (Beginner | Intermediate | Advanced | AllLevel)
        model_config: Optional dict from .NET backend. If provided and valid,
                      uses create_llm_from_config(); otherwise falls back to create_llm().

    Returns:
        dict with keys `nodes` (list) and `edges` (list).

    Raises:
        ValueError  : If the LLM returns invalid JSON.
        LLMProviderError: If the LLM call itself fails.
    """
    with timer("Roadmap LLM generation"):
        # Choose LLM: prefer backend config if available, else .env singleton
        if model_config and model_config.get("api_base"):
            llm = create_llm_from_config(model_config)
            logger.info("Using backend-provided model_config for roadmap generation")
        else:
            llm = create_llm()
            logger.info("Using default LLM provider for roadmap generation")

        messages = [
            ("system", ROADMAP_SYSTEM_PROMPT),
            ("human", ROADMAP_USER_PROMPT.format(topic=topic, level=level)),
        ]

        response = llm.invoke(messages)
        raw = response.content.strip()

        # Strip markdown code fences if LLM wraps JSON in ```json ... ```
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"LLM returned invalid JSON: {raw[:200]}")
            raise ValueError(f"LLM returned invalid JSON: {e}") from e

    return data
