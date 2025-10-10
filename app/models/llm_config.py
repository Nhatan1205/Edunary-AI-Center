import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

def get_llm():
    """
    Get LLM instance based on environment configuration.
    
    Set LLM_PROVIDER in .env file:
    - 'ollama' (default): Use local Ollama
    - 'gemini': Use Google Gemini
    - 'huggingface': Use Hugging Face Router API
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    if provider == "gemini":
        # Google Gemini configuration
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )
    
    elif provider == "huggingface":
        # Hugging Face Router API configuration
        # Uses OpenAI-compatible API with HF Router
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN not found in environment variables")
        
        model_name = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        
        return ChatOpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token,
            model=model_name,
            temperature=0.7,
            max_tokens=512
        )
    
    else:  # Default to Ollama
        model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
        return OllamaLLM(model=model_name)

# Initialize LLM instance
llm = get_llm()
