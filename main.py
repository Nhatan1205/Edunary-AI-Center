from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.routes import chat
from app.routes import websocket_chat
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI Chatbot API",
    description="AI-powered chatbot with web interface",
    version="1.0.0"
)

# Setup templates
templates = Jinja2Templates(directory="app/templates")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include existing API routes
app.include_router(chat.router, prefix="/api")

# Include WebSocket routes
app.include_router(websocket_chat.router)

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface(request: Request):
    """Serve the chat interface"""
    return templates.TemplateResponse("chat_new.html", {"request": request})

@app.get("/embed", response_class=HTMLResponse)
async def get_embeddable_chat(request: Request):
    """Serve embeddable chat interface for iframe"""
    return templates.TemplateResponse("chat_new.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# uvicorn main:app --reload
# Chat interface: http://127.0.0.1:8000/
# API docs: http://127.0.0.1:8000/docs
# API chat: http://127.0.0.1:8000/api/chat/