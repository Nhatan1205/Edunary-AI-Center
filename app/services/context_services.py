from langchain_mongodb import MongoDBChatMessageHistory
from langchain.schema import messages_to_dict, messages_from_dict

# Thông tin kết nối Mongo
MONGO_URI = "mongodb+srv://nguyenmaihuyhoang_db_user:K8czZk9gtvsO9UqA@cluster0.tg51r0c.mongodb.net/"
DB_NAME = "chatbot"
COLLECTION_NAME = "chat_sessions"

def _get_history_object(session_id: str) -> MongoDBChatMessageHistory:
    """Create a history object for session_id"""
    return MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=MONGO_URI,
        database_name=DB_NAME,
        collection_name=COLLECTION_NAME,
    )
    
def get_session_history(session_id: str):
    """
    Get the entire conversation history (as a list dict)
    """
    history = _get_history_object(session_id)
    messages = history.messages
    return messages_to_dict(messages)  # Trả về list[dict]

def save_message(session_id: str, role: str, message: str):
    """
    Save a message to MongoDB via MongoDBChatMessageHistory
    role: "user" or "assistant"
    """
    history = _get_history_object(session_id)

    if role == "user":
        history.add_user_message(message)
    elif role == "assistant":
        history.add_ai_message(message)
    else:
        raise ValueError("Role must be 'user' or 'assistant'")