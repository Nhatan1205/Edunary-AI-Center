from app.services.query_data import query_rag
from langchain_core.prompts import PromptTemplate
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

entity_search_prompt = PromptTemplate.from_template("""
You are an Entity Extraction System for an education platform.
Your goal is to identify the core **Subject** or **Topic** the user wants to learn.

INSTRUCTIONS:
1. Analyze the user's input to find the main keyword (Technology, Skill, or Instructor name).
2. Ignore stop words like "I want to learn", "course", "tutorial", "best", "online".
3. Return the result in valid JSON format with a single key "topic".
4. If no specific topic is found (e.g., user says "hello"), return "topic": null.

EXAMPLES:
Input: "I want to find a Python course for beginners"
Output: {{ "topic": "Python" }}

Input: "Show me courses about Digital Marketing"
Output: {{ "topic": "Digital Marketing" }}

Input: "Do you have any ReactJS tutorials?"
Output: {{ "topic": "ReactJS" }}

Input: "Courses by Le Van Vinh"
Output: {{ "topic": "Le Van Vinh" }}

Conversation:
{chat_history}
User: {input}

Output JSON:
""")

summary_prompt = PromptTemplate.from_template("""
You are a helpful academic assistant for Edunary.
The user asked: "{input}"

Here is the list of courses found in the database (JSON format):
{courses_json}

INSTRUCTIONS:
- If the list is empty, apologize and suggest user to try another keyword.
- If courses are found, summarize them naturally (Title, Instructor, Price).
- Do NOT list JSON raw data.
- Do NOT mention "backend", "API", or "database".
- Be concise and encouraging.
""")

def handle_greeting(message: str) -> dict:
    return {
        "reply": "Hi! Nice to meet you 👋 How can I help you find the perfect course today?",
        "messageType": "text",
        "courses": None
    }

def handle_faq(message: str) -> dict:
    text_reply = query_rag(message)
    return {
        "reply": text_reply,
        "messageType": "faq",
        "courses": None
    }

def handle_get_course_info(message: str, history: str) -> dict:
    topic = ""
    try:
        raw_result = (entity_search_prompt | llm).invoke({"chat_history": history, "input": message})
        
        result_content = raw_result.content if hasattr(raw_result, 'content') else raw_result
        clean_json = clean_json_response(result_content)
        
        entity_data = json.loads(clean_json)
        topic = entity_data.get("topic", "")
        
        if not topic:
            return {
                "reply": "I am not sure which topic you are looking for. Could you please specify a subject (e.g., Python, Marketing, Web Development)?",
                "messageType": "text",
                "courses": None
            }

    except json.JSONDecodeError:
        return {
            "reply": "Sorry, the system is busy. Please try again later.",
            "messageType": "text",
            "courses": None
        }
    except Exception:
        return {
            "reply": "An error occurred while processing your request.",
            "messageType": "text",
            "courses": None
        }

    api_url = "https://edunary.runasp.net/api/Courses/search"
    
    payload = {
        "searchText": topic,
        "pageNumber": 1,
        "pageSize": 5,
        "sortBy": 0,
        "filterData": []
    }
    try:
        response = requests.post(api_url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get("items", [])
            
            if not items:
                return {
                    "reply": f"I searched for '{topic}' but could not find any matching courses. Try searching for 'Python', 'Marketing', or 'Design'.",
                    "messageType": "text",
                    "courses": None
                }

            # Prepare data for LLM summary (all courses)
            simplified_courses = []
            for item in items:
                simplified_courses.append({
                    "title": item.get("title"),
                    "instructor": item.get("instructorName"),
                    "price": item.get("price")
                })
            
            courses_str = json.dumps(simplified_courses, ensure_ascii=False)
            
            # Generate summary using LLM
            summary_res = (summary_prompt | llm).invoke({
                "chat_history": history, 
                "input": message, 
                "courses_json": courses_str
            })
            
            summary_text = (summary_res.content if hasattr(summary_res, 'content') else summary_res).strip()
            
            # Build simplified course card for ONLY the first course
            first_item = items[0]
            course_card = {
                "id": first_item.get("id"),
                "title": first_item.get("title", "Untitled Course"),
                "subtitle": first_item.get("subtitle"),
                "imageUrl": first_item.get("imageUrl"),
                "instructorName": first_item.get("instructorName", "Unknown"),
                "courseUrl": f"https://edunary.runasp.net/course/{first_item.get('id')}"
            }
            
            return {
                "reply": summary_text,
                "messageType": "course_cards",
                "courses": [course_card]  # Only first course
            }
            
        else:
            return {
                "reply": "Sorry, I cannot connect to the course database right now.",
                "messageType": "text",
                "courses": None
            }

    except requests.RequestException:
        return {
            "reply": "The system is currently undergoing maintenance. Please check back later.",
            "messageType": "text",
            "courses": None
        }

def handle_default(message: str) -> str:
    return None
