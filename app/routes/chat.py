from fastapi import APIRouter, HTTPException
from app.models.ChatResponse import ChatResponse
from app.models.ChatRequest import ChatRequest
from app.services.chat_services import process_message
from app.models.llm_config import llm
import uuid
router = APIRouter(prefix="/chat", tags=["chat"])   

@router.post("/", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        result = process_message(session_id, request.message)
        
        return ChatResponse(
            session_id=session_id,
            reply=result["reply"],
            messageType=result["messageType"],
            courses=result.get("courses")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))