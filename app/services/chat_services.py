from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
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
You are an intent classifier for an e-learning platform called "Edunary".
Your job is to determine the user's final intent based on the conversation history and the latest input.

Analyze the user's latest message (User input) in the context of the history:

1. **'greeting'**:
   - Usage: User sends social pleasantries (e.g., "Hello", "Hi", "Good morning", "Are you there?").

2. **'get_course_information'**:
   - Usage: User wants to **find, filter, or know details** about courses.
   - Triggers:
     - Searching for a topic (e.g., "Do you have Python courses?", "I want to learn React").
     - Asking for recommendations (e.g., "What course is good for beginners?", "Suggest a web dev course").
     - Asking specific details about a course (Price, Instructor, Duration, Curriculum, Rating).
     - Comparing courses.

3. **'search_faq'**:
   - Usage: User asks about the **platform's functionality or policies** (NOT specific course content).
   - Triggers: "How do I register?", "Payment methods", "Forgot password", "Refund policy", "Contact support", "About us".

4. **'fallback'**:
   - Usage: User talks about unrelated topics (Weather, Sports, Jokes) or the input makes no sense.

INSTRUCTIONS:
- Give more weight to the most recent message.
- If the user asks a mixed question (e.g., "Hi, do you have Java courses?"), prioritize the specific intent ('get_course_information') over 'greeting'.
- Respond with ONLY ONE word from the list: ['greeting', 'get_course_information', 'search_faq', 'fallback'].
- Do NOT output explanations.

Conversation History:
{chat_history}

User (most recent): {input}

Intent:
""")


def process_message(session_id: str, message: str) -> dict:
    # Build history text
    history_data = get_session_history(session_id)
    recent_history = history_data[-10:]
    history = ""
    for msg in recent_history:
        role = msg["type"]
        role_name = "user" if role == "human" else "assistant"
        history += f"{role_name}: {msg['data']['content']}\n"
    
    # Detect intent
    intent_result = (intent_prompt | llm).invoke({"chat_history": history, "input": message})
    intent = (intent_result.content if hasattr(intent_result, 'content') else intent_result).strip().lower()

    # Handle by intent - all handlers now return dict
    if "greeting" in intent:
        result = handle_greeting(message)
    elif "search_faq" in intent:
        result = handle_faq(message)
    elif "get_course_information" in intent:
        result = handle_get_course_info(message, history)
    else:
        result = {
            "reply": "Sorry, I can only answer questions about Edunary courses and platform features. How can I help you today?",
            "messageType": "text",
            "courses": None
        }

    # Persist conversation (save only text reply to history)
    save_message(session_id, "user", message)
    save_message(session_id, "assistant", result["reply"])

    return result