"""
In-memory rate limiter for WebSocket connections.

Limits the number of messages a single session can send within a time window.
No external dependency — suitable for single-instance deployments.
"""

import time
from collections import defaultdict, deque


class RateLimiter:
    """
    Sliding-window rate limiter keyed by session_id.

    Args:
        max_requests: Max messages allowed within the window.
        window_seconds: Duration of the sliding window in seconds.
    """

    def __init__(self, max_requests: int = 15, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # session_id → deque of timestamps
        self._timestamps: dict[str, deque] = defaultdict(deque)

    def allow(self, session_id: str) -> bool:
        """
        Check if session_id is within the rate limit.

        Returns True if the request is allowed, False if rate-limited.
        """
        now = time.monotonic()
        window_start = now - self.window_seconds
        queue = self._timestamps[session_id]

        # Drop timestamps outside the window
        while queue and queue[0] < window_start:
            queue.popleft()

        if len(queue) >= self.max_requests:
            return False

        queue.append(now)
        return True
