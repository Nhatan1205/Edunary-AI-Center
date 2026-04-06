"""
Chat history store — MongoDB-backed conversation persistence.
"""

from pymongo import MongoClient
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_core.messages import messages_to_dict

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Create MongoClient at startup — reused across all requests
_settings = get_settings()
_mongo_client = MongoClient(_settings.MONGO_URI)
logger.info("MongoDB client initialized")


def _get_history_object(session_id: str) -> MongoDBChatMessageHistory:
    return MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string="",
        client=_mongo_client,
        database_name=_settings.MONGO_DB_NAME,
        collection_name=_settings.MONGO_COLLECTION_NAME,
    )


def get_session_history(session_id: str) -> list[dict]:
    history = _get_history_object(session_id)
    return messages_to_dict(history.messages)


def save_message(session_id: str, role: str, message: str) -> None:
    history = _get_history_object(session_id)
    if role == "user":
        history.add_user_message(message)
    elif role == "assistant":
        history.add_ai_message(message)
    else:
        raise ValueError("Role must be 'user' or 'assistant'")
