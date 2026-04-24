"""
WebSocket controller — WebSocket endpoint for real-time chat.
Replaces routes/websocket_chat.py.
No authentication on WebSocket (as requested).
"""

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.chat_service import process_message
from app.services.websocket_manager import manager, stream_response
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat.

    Args:
        websocket: WebSocket connection.
        session_id: Unique session identifier from client.
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
                result = process_message(session_id, user_message)

                # Stream response word by word
                await stream_response(result["reply"], session_id)

                # Send course cards if present
                if result.get("messageType") == "course_cards" and result.get("courses"):
                    await manager.send_message(json.dumps({
                        "type": "course_cards",
                        "courses": result["courses"]
                    }), session_id)

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(session_id)
