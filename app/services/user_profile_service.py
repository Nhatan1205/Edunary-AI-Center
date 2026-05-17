"""
User Profile Service — retrieves a user's profile from the Qdrant `edunary_users` collection.

Used by the chat service to personalise responses for authenticated users.

Key design decisions:
  - No in-memory cache: user profiles are stable, and a single chat session is short-lived.
    If the user needs fresh data they can reload the chatbot widget.
  - Returns None (not an error) when the profile is absent, so the caller can gracefully
    fall back to the anonymous chat flow.
  - Reuses the same UUID5 scheme as `user_embedding_service` so point IDs are identical.
"""

import uuid
from dataclasses import dataclass, field
from typing import Optional

from app.embeddings.factory import build_qdrant_client
from app.core.logging import get_logger

logger = get_logger(__name__)

USER_COLLECTION = "edunary_users"
_UUID_NAMESPACE = uuid.NAMESPACE_DNS


@dataclass
class UserProfileContext:
    """Lightweight user context injected into chat prompts."""

    user_id: str
    user_name: str
    full_name: Optional[str] = None
    goal: Optional[str] = None
    skill_level: Optional[str] = None
    preferred_categories: list[str] = field(default_factory=list)
    preferred_topics: list[str] = field(default_factory=list)
    weekly_hours: Optional[int] = None

    @property
    def display_name(self) -> str:
        """Return the best available name for greeting purposes."""
        return self.full_name or self.user_name

    def to_prompt_text(self) -> str:
        """
        Render the profile as a compact text block for injection into LLM prompts.
        Only includes fields that have meaningful values to keep prompts concise.
        """
        lines: list[str] = [f"User: {self.display_name}"]
        if self.goal:
            lines.append(f"Learning goal: {self.goal}")
        if self.skill_level:
            lines.append(f"Skill level: {self.skill_level}")
        if self.preferred_categories:
            lines.append("Preferred categories: " + ", ".join(self.preferred_categories))
        if self.preferred_topics:
            lines.append("Preferred topics: " + ", ".join(self.preferred_topics))
        if self.weekly_hours is not None:
            lines.append(f"Weekly study hours: {self.weekly_hours}")
        return "\n".join(lines)


def _user_point_id(user_id: str) -> str:
    """Mirror of user_embedding_service._user_point_id — must stay in sync."""
    return str(uuid.uuid5(_UUID_NAMESPACE, f"edunary_user:{user_id}"))


def get_user_profile(user_id: str) -> Optional[UserProfileContext]:
    """
    Retrieve a user's profile payload from Qdrant.

    Args:
        user_id: ASP.NET Identity GUID string.

    Returns:
        UserProfileContext if the user has been embedded, None otherwise.
    """
    try:
        client, _ = build_qdrant_client()

        if not client.collection_exists(USER_COLLECTION):
            logger.debug(f"Collection '{USER_COLLECTION}' does not exist — returning anonymous.")
            return None

        point_id = _user_point_id(user_id)
        results = client.retrieve(
            collection_name=USER_COLLECTION,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )

        if not results:
            logger.info(f"User {user_id} not found in Qdrant — falling back to anonymous mode.")
            return None

        payload = results[0].payload or {}

        profile = UserProfileContext(
            user_id=user_id,
            user_name=payload.get("user_name", ""),
            full_name=payload.get("full_name"),
            goal=payload.get("goal"),
            skill_level=payload.get("skill_level"),
            preferred_categories=payload.get("preferred_categories") or [],
            preferred_topics=payload.get("preferred_topics") or [],
            weekly_hours=payload.get("weekly_hours"),
        )

        logger.info(f"Loaded profile for user {user_id} — '{profile.display_name}'")
        return profile

    except Exception as e:
        # Never crash the chat pipeline due to a profile lookup failure.
        logger.warning(f"Could not retrieve profile for user {user_id}: {e}")
        return None
