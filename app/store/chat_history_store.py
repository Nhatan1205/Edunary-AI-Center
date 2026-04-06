"""
Chat history store — MongoDB-backed conversation persistence.
Refactored from services/context_services.py.
"""

from langchain_mongodb import MongoDBChatMessageHistory
from langchain_core.messages import messages_to_dict

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _get_history_object(session_id: str) -> MongoDBChatMessageHistory:
    """Create a MongoDBChatMessageHistory object for the given session."""
    settings = get_settings()
    return MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=settings.MONGO_URI,
        database_name=settings.MONGO_DB_NAME,
        collection_name=settings.MONGO_COLLECTION_NAME,
    )


def get_session_history(session_id: str) -> list[dict]:
    """
    Get the entire conversation history as a list of dicts.

    Returns:
        list[dict]: Each dict has 'type' ('human'/'ai') and 'data' with 'content'.
    """
    history = _get_history_object(session_id)
    messages = history.messages
    return messages_to_dict(messages)


def save_message(session_id: str, role: str, message: str) -> None:
    """
    Save a message to MongoDB.

    Args:
        session_id: Chat session identifier.
        role: Either 'user' or 'assistant'.
        message: Message content.
    """
    history = _get_history_object(session_id)

    if role == "user":
        history.add_user_message(message)
    elif role == "assistant":
        history.add_ai_message(message)
    else:
        raise ValueError("Role must be 'user' or 'assistant'")
