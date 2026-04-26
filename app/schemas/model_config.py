"""
Shared ModelConfig schema.

The .NET backend always sends model configuration in the request body
under the key `model_config`. Every AI endpoint that needs an LLM
should accept this schema.

This mirrors the pattern in CallFlow Drive where model_config is built
from the SystemSettings table and passed alongside the actual request payload.
"""

from pydantic import BaseModel
from typing import Optional


class ModelConfig(BaseModel):
    """
    LLM provider config sent by the .NET backend.

    The backend reads these values from the SystemSettings table
    (keys: LLM_ModelName, LLM_ApiKey, LLM_BaseUrl) and passes them here,
    so the AI Center never needs to know which provider is active —
    it just uses whatever config it receives.

    Fields:
        model_name  : Model identifier (e.g. "gpt-4o", "gemini-2.5-flash")
        api_key     : Provider API key
        api_base    : Base URL of the provider (must be OpenAI-compatible)
        api_version : Optional version string (e.g. Azure OpenAI)
        temperature : Sampling temperature (default 0.7)
        max_tokens  : Max output tokens (default 512)
    """

    model_name: str = ""
    api_key: str = ""
    api_base: str = ""
    api_version: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512
