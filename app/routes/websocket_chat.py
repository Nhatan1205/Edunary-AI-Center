from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.chat_services import process_message
import json
import asyncio

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_message(self, message: str, session_id: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(message)
    
    async def send_streaming_start(self, session_id: str):
        """Send signal to start streaming"""
        await self.send_message(json.dumps({
            "type": "stream_start"
        }), session_id)
    
    async def send_streaming_chunk(self, content: str, session_id: str):
        """Send a chunk of streaming content"""
        await self.send_message(json.dumps({
            "type": "stream_chunk",
            "content": content
        }), session_id)
    
    async def send_streaming_end(self, session_id: str):
        """Send signal to end streaming"""
        await self.send_message(json.dumps({
            "type": "stream_end"
        }), session_id)

manager = ConnectionManager()

async def stream_response(response_text: str, session_id: str, delay: float = 0.03):
    """
    Stream response word by word to create typing effect
    
    Args:
        response_text: The full response text to stream
        session_id: Session ID to send to
        delay: Delay between each word in seconds (default 0.03s = 30ms)
    """
    await manager.send_streaming_start(session_id)
    
    # Split response into words for more natural streaming
    words = response_text.split(' ')
    
    for i, word in enumerate(words):
        # Add space after word except for the last one
        chunk = word + (' ' if i < len(words) - 1 else '')
        await manager.send_streaming_chunk(chunk, session_id)
        await asyncio.sleep(delay)
    
    await manager.send_streaming_end(session_id)

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat
    
    Args:
        websocket: WebSocket connection
        session_id: Unique session identifier from client
    """
    await manager.connect(websocket, session_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            
            if user_message:
                # Process message using chat service
                reply = process_message(session_id, user_message)
                
                # Stream response word by word
                await stream_response(reply, session_id)
                
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        manager.disconnect(session_id)
