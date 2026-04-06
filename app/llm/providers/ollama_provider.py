"""Ollama local LLM provider."""

from langchain_ollama import OllamaLLM
from app.core.config import Settings


def create_ollama(settings: Settings):
    """Create a local OllamaLLM instance."""
    return OllamaLLM(model=settings.OLLAMA_MODEL)
