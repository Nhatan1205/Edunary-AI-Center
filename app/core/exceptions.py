"""
Custom exception classes for the Edunary AI Center.
These are caught by the global exception handler in main.py.
"""


class EdunaryAIException(Exception):
    """Base exception for all Edunary AI errors."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class LLMProviderError(EdunaryAIException):
    """Raised when an LLM call fails."""

    def __init__(self, message: str = "LLM provider error"):
        super().__init__(message, 500)


class BackendAPIError(EdunaryAIException):
    """Raised when the Edunary .NET backend API call fails."""

    def __init__(self, message: str = "Backend API error"):
        super().__init__(message, 502)


class AuthenticationError(EdunaryAIException):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Not authenticated"):
        super().__init__(message, 401)
