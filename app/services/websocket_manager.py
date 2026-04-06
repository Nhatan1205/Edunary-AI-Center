"""
WebSocket connection manager.
Handles active WebSocket connections and streaming responses.
Extracted from routes/websocket_chat.py.
"""

import json
import asyncio

from fastapi import WebSocket

from app.core.logging import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections by session ID."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept and register a WebSocket connection."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected: session={session_id}")

    def disconnect(self, session_id: str):
        """Remove a WebSocket connection."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected: session={session_id}")

    async def send_message(self, message: str, session_id: str):
        """Send a text message to a specific session."""
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(message)

    async def send_streaming_start(self, session_id: str):
        """Send signal to start streaming."""
        await self.send_message(json.dumps({
            "type": "stream_start"
        }), session_id)

    async def send_streaming_chunk(self, content: str, session_id: str):
        """Send a chunk of streaming content."""
        await self.send_message(json.dumps({
            "type": "stream_chunk",
            "content": content
        }), session_id)

    async def send_streaming_end(self, session_id: str):
        """Send signal to end streaming."""
        await self.send_message(json.dumps({
            "type": "stream_end"
        }), session_id)


# Singleton instance
manager = ConnectionManager()


async def stream_response(response_text: str, session_id: str, delay: float = 0.03):
    """
    Stream response word by word to create typing effect.

    Args:
        response_text: The full response text to stream.
        session_id: Session ID to send to.
        delay: Delay between each word in seconds (default 0.03s = 30ms).
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
