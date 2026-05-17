"""
Edunary AI Center — FastAPI Application Entry Point.

This is the main entry point for the application.
It registers middleware, exception handlers, and all controllers (routers).
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.core.exceptions import EdunaryAIException
from app.core.logging import setup_logging, get_logger

# Setup logging FIRST — before any other imports that may log at module level
settings = get_settings()
setup_logging(debug=settings.DEBUG)
logger = get_logger(__name__)

# Controllers imported AFTER logging is ready
from app.controllers.chat_controller import router as chat_router
from app.controllers.websocket_controller import router as ws_router
from app.controllers.roadmap_controller import router as roadmap_router
from app.controllers.quiz_controller import router as quiz_router
from app.controllers.rag_controller import router as rag_router
from app.controllers.qdrant_dashboard_controller import router as qdrant_dashboard_router
from app.controllers.course_embedding_controller import router as course_embedding_router
from app.controllers.user_embedding_controller import router as user_embedding_router
from app.controllers.transcription_controller import router as transcription_router

# Create FastAPI app
app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description="Edunary AI Center — Chatbot, Roadmap Generator, Quiz Creator",
)

# Setup templates
templates = Jinja2Templates(directory="app/templates")

# Mount static files (CSS, JS extracted from templates)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://localhost:44447",
        "https://edunary.runasp.net"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Global Exception Handler ===
@app.exception_handler(EdunaryAIException)
async def edunary_exception_handler(request: Request, exc: EdunaryAIException):
    """Handle all custom Edunary exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "message": exc.message},
    )


# === Register Controllers ===
app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])
app.include_router(ws_router, tags=["WebSocket"])
app.include_router(roadmap_router, prefix="/api/roadmap", tags=["Roadmap"])
app.include_router(rag_router, prefix="/api/rag", tags=["RAG"])
app.include_router(qdrant_dashboard_router, prefix="/api/qdrant", tags=["Qdrant Dashboard"])
app.include_router(course_embedding_router, prefix="/api/course-embeddings", tags=["Course Embeddings"])
app.include_router(user_embedding_router, prefix="/api/user-embeddings", tags=["User Embeddings"])
app.include_router(quiz_router, prefix="/api/quiz", tags=["Quiz"])
app.include_router(transcription_router, prefix="/api/transcription", tags=["Transcription"])


# === Health Check ===
@app.get("/health")
async def health_check():
    """Health check endpoint — returns app version and current LLM configuration."""
    provider = settings.LLM_PROVIDER.lower()

    # Determine which model name is active based on provider
    model_map = {
        "gemini": settings.GEMINI_MODEL,
        "huggingface": settings.HUGGINGFACE_MODEL,
        "ollama": settings.OLLAMA_MODEL,
        "custom": settings.CUSTOM_LLM_MODEL or "custom",
    }

    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "llm": {
            "provider": provider,
            "model": model_map.get(provider, "unknown"),
        },
        "embedding": {
            "provider": settings.EMBEDDING_PROVIDER,
        },
        "vector_store": {
            "provider": "qdrant",
            "url": settings.QDRANT_URL,
            "collection": settings.QDRANT_COLLECTION,
        },
        "auth_enabled": settings.AUTH_ENABLED,
    }


# === Demo HTML Routes ===
@app.get("/", response_class=HTMLResponse)
async def get_chat_interface(request: Request):
    """Serve the chat interface."""
    return templates.TemplateResponse(
        request=request,
        name="chat_new.html",
    )


@app.get("/embed", response_class=HTMLResponse)
async def get_embeddable_chat(request: Request):
    """Serve embeddable chat interface for iframe."""
    return templates.TemplateResponse(
        request=request,
        name="chat_new.html",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

# uvicorn main:app --reload
# Chat interface: http://127.0.0.1:8000/
# API docs: http://127.0.0.1:8000/docs
# API chat: http://127.0.0.1:8000/api/chat/
# Health: http://127.0.0.1:8000/health
# & "D:\tailieu4\HK1-25-26\New Technology\Demo\AI Backend\chatbot\Scripts\Activate.ps1"
# python -m app.store.document_loader