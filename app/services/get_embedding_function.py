import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()


def get_embedding_function():
    """
    Get embedding function based on environment configuration.
    
    Set EMBEDDING_PROVIDER in .env file:
    - 'ollama' (default): Use local Ollama embeddings
    - 'openai': Use OpenAI embeddings
    - 'huggingface': Use Hugging Face Inference API (online, no download)
    - 'google': Use Google Generative AI embeddings
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()
    
    if provider == "openai":
        # OpenAI embeddings
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(
            model=model,
            openai_api_key=api_key
        )
    
    elif provider == "huggingface":
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN not found in environment variables")
        
        model = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEndpointEmbeddings(
            model=model,
            huggingfacehub_api_token=hf_token
        )
    
    elif provider == "google":
        # Google Generative AI embeddings
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
        return GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=api_key
        )
    
    else:  # Default to Ollama
        model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        return OllamaEmbeddings(model=model)
