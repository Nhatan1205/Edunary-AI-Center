from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from app.services.intent_handlers import handle_greeting, handle_faq, handle_get_course_info
from app.services.context_services import get_session_history, save_message
from app.models.llm_config import llm

# Prompts
chat_prompt = PromptTemplate.from_template("""
This is a conversation between User and Assistant.
{chat_history}
User: {input}
Assistant:
""")

intent_prompt = PromptTemplate.from_template("""
Based on the entire conversation below, determine the user's final intent.
Give more weight to the most recent messages.

Only respond with ONE word from the list below. Do NOT explain, do NOT add anything else, and respond with exactly one word:
- If the user is greeting (hello, hi, hey, good morning, etc.) => 'greeting'
- If the user is asking about course-related information (course details, price, duration, instructor, category, level, enrollment, etc.) => 'get_course_information'
- If the user is asking about general platform information (about us, policies, how to use the platform, contact info, terms of service, privacy policy, company information, etc.) => 'search_faq'
- If the user's question is NOT related to the e-learning platform at all (random topics, weather, sports, unrelated questions) => 'fallback'
- If you are uncertain about the user's intent => 'fallback'

Conversation:
{chat_history}
User (most recent): {input}

Intent:
""")


def process_message(session_id: str, message: str) -> str:

    # Build history text
    history_data = get_session_history(session_id)
    recent_history = history_data[-10:]
    history = ""
    for msg in recent_history:
        role = msg["type"]
        role_name = "user" if role == "human" else "assistant"
        history += f"{role_name}: {msg['data']['content']}\n"

    # reply = (chat_prompt | llm).invoke({"chat_history": history, "input": message})
    # save_message(session_id, "user", message)
    # save_message(session_id, "assistant", reply)
    
    
    # Detect intent
    intent_result = (intent_prompt | llm).invoke({"chat_history": history, "input": message})
    # Handle both string (OllamaLLM) and AIMessage (ChatOpenAI, ChatGoogleGenerativeAI)
    intent = (intent_result.content if hasattr(intent_result, 'content') else intent_result).strip().lower()

    # Handle by intent
    if "greeting" in intent:
        reply = handle_greeting(message)
    elif "search_faq" in intent:
        reply = handle_faq(message)
    # elif "get_course_information" in intent:
    #     reply = handle_get_course_info(message, history)
    else:
        reply = "Sorry, I can only answer questions about this website. Can you ask something related to our features or services?"

    # Persist conversation
    save_message(session_id, "user", message)
    save_message(session_id, "assistant", reply)

    return reply

# Prompts
chat_prompt = PromptTemplate.from_template("""
This is a conversation between User and Assistant.
{chat_history}
User: {input}
Assistant:
""")

intent_prompt = PromptTemplate.from_template("""
Based on the entire conversation below, determine the user's final intent.
Give more weight to the most recent messages.

Only respond with ONE word from the list below. Do NOT explain, do NOT add anything else, and respond with exactly one word:
- If the user’s question is NOT related to movie ticket booking or static website information => 'fallback'.
- If the user is greeting => 'greeting'
- If the user is asking about any movie-related information, including title, description, genre and nothing else => 'get_movie_information'
- If the user is asking a FAQ about static information related to movie ticket booking, such as About Us, site policies, usage instructions, or other website services (not including questions about specific movies or showtimes) => 'search_faq'
- If you are not certain that the user's intent matches any of the above => 'fallback'

Conversation:
{chat_history}
User (most recent): {input}

Intent:
""")


def process_message(session_id: str, message: str) -> str:

    # Build history text
    history_data = get_session_history(session_id)
    recent_history = history_data[-10:]
    history = ""
    for msg in recent_history:
        role = msg["type"]
        role_name = "user" if role == "human" else "assistant"
        history += f"{role_name}: {msg['data']['content']}\n"

    # reply = (chat_prompt | llm).invoke({"chat_history": history, "input": message})
    # save_message(session_id, "user", message)
    # save_message(session_id, "assistant", reply)
    
    
    # Detect intent
    intent_result = (intent_prompt | llm).invoke({"chat_history": history, "input": message})
    # Handle both string (OllamaLLM) and AIMessage (ChatOpenAI, ChatGoogleGenerativeAI)
    intent = (intent_result.content if hasattr(intent_result, 'content') else intent_result).strip().lower()

    # Handle by intent
    if "greeting" in intent:
        reply = handle_greeting(message)
    elif "search_faq" in intent:
        reply = handle_faq(message)
    elif "get_course_information" in intent:
        reply = handle_get_course_info(message, history)
    else:
        reply = "Sorry, I can only answer questions about this website. Can you ask something related to our features or services?"

    # Persist conversation
    save_message(session_id, "user", message)
    save_message(session_id, "assistant", reply)

    return reply