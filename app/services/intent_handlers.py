from app.services.query_data import query_rag
from langchain.prompts import PromptTemplate
from app.models.llm_config import llm
import requests
import json
import re

def clean_json_response(text: str) -> str:
    """
    Clean JSON response from LLM by removing markdown code blocks.
    
    Handles formats like:
    - ```json\n{...}\n```
    - ```\n{...}\n```
    - Just {...}
    """
    text = text.strip()
    
    # Remove markdown code blocks
    # Pattern: ```json or ``` at start, ``` at end
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    
    return text.strip()

entity_prompt = PromptTemplate.from_template("""
You are an information extraction tool. ALWAYS return valid JSON.
DO NOT add any other text or markdown.
DO NOT add ``` marks or explanations.

Extract information from the user's query and return JSON with the following fields:
{{
  "title": string,  # course title
  "category": [string], # array of course categories (e.g., programming, business, design)
  "instructor": string, # instructor name
  "level": string, # course level (beginner, intermediate, advanced)
  "price": number, # course price (leave null if not specified)
  "duration": number  # course duration in hours
}}

IMPORTANT NOTES:
- Use empty string ("") for string, [] for array, null for number if no information is available
- DO NOT extract question words (how much, how long) as values
- price and duration must be numbers
- category is an array of strings
- level must be one of: "beginner", "intermediate", "advanced", or ""

Examples:
Input: "How long is the Python course?"
Output: {{"title": "Python", "category": [], "instructor": "", "level": "", "price": null, "duration": null}}

Input: "How much does Web Development cost?"
Output: {{"title": "Web Development", "category": [], "instructor": "", "level": "", "price": null, "duration": null}}

Input: "advanced JavaScript course by John Doe"
Output: {{"title": "JavaScript", "category": ["programming"], "instructor": "John Doe", "level": "advanced", "price": null, "duration": null}}

Conversation:
{chat_history}
User: {input}
""")



summary_prompt = PromptTemplate.from_template("""
You are an assistant that answers questions about online courses on the Edunary e-learning platform.  

Conversation:
{chat_history}
User (most recent): {input}

Here is the raw course data (JSON):
{courses_json}

Instructions:
- Provide a clear, natural, and concise response.  
- Carefully read the user's most recent question and answer it using only the relevant information from the platform data. Do not write any words relate to "platform" or "backend"  
- If the question is broad or general, you may provide a more complete summary of the course (title, description, duration, category, level, price, instructor).  
- Always make sure the response directly addresses the user's intent, without adding unrelated details.
""")

def handle_greeting(message: str) -> str:
    return "Hi! Nice to meet you 👋"

def handle_faq(message: str) -> str:
    return query_rag(message)

# def handle_get_course_info(message: str, history: str) -> str:
#     # Extract entities
#     entities_result = (entity_prompt | llm).invoke({"chat_history": history, "input": message})
#     # Handle both string (OllamaLLM) and AIMessage (ChatOpenAI, ChatGoogleGenerativeAI)
#     entities_str = (entities_result.content if hasattr(entities_result, 'content') else entities_result).strip()

#     # Clean markdown code blocks from JSON response
#     entities_str_clean = clean_json_response(entities_str)

#     try:
#         entities = json.loads(entities_str_clean)
        
#     except json.JSONDecodeError as e:
#         return "Sorry, I couldn't extract any course information. Could you rephrase your question?"

#     # Filter out invalid fields (remove "none", "null", empty lists)
#     valid_entities = {}
#     for k, v in entities.items():
#         if isinstance(v, list):
#             clean_list = [item for item in v if item not in ["none", "null", "", '']]
#             if clean_list:
#                 valid_entities[k] = clean_list
#         elif v not in [None, "none", "null", "", '']:
#             valid_entities[k] = v
    

#     # Fallback if no valid entities
#     if not valid_entities:
#         return "Sorry, I couldn't find any course information from your message. Please ask about a course."

#     # Call backend API with only valid entities
#     try:
#         response = requests.get("http://localhost:5000/course/search", json=valid_entities)
#         response.raise_for_status()
#         # parse response JSON
#         api_response = response.json()
#         courses_str = json.dumps(api_response, ensure_ascii=False)

#         # let LLM summarize
#         summary_result = (summary_prompt | llm).invoke({"chat_history": history, "input": message, "courses_json": courses_str})
#         # Handle both string (OllamaLLM) and AIMessage (ChatOpenAI, ChatGoogleGenerativeAI)
#         summary = (summary_result.content if hasattr(summary_result, 'content') else summary_result).strip()
#         return summary
#     except requests.RequestException as e:
#         return "Sorry, I failed to fetch course information."

def handle_default(message: str) -> str:
    return None  # để fallback sang chatbot LLM
