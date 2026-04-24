"""
Backend API client — HTTP client for calling the Edunary .NET Backend.
Replaces hardcoded requests.post("https://edunary.runasp.net/...") calls.
"""

import requests

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.exceptions import BackendAPIError

logger = get_logger(__name__)


class BackendAPIClient:
    """HTTP client wrapper for the Edunary .NET backend API."""

    def __init__(self):
        settings = get_settings()
        self.base_url = settings.BACKEND_API_BASE_URL.rstrip("/")
        self.timeout = settings.BACKEND_API_TIMEOUT

    def search_courses(
        self,
        search_text: str,
        page: int = 1,
        page_size: int = 5,
        sort_by: int = 0,
    ) -> dict:
        """
        Search courses via POST /api/Courses/search.

        Args:
            search_text: Topic/keyword to search.
            page: Page number.
            page_size: Number of results per page.
            sort_by: Sort order (0 = relevant).

        Returns:
            dict: API response with 'items' list.

        Raises:
            BackendAPIError: If the API call fails.
        """
        url = f"{self.base_url}/api/Courses/search"
        payload = {
            "searchText": search_text,
            "pageNumber": page,
            "pageSize": page_size,
            "sortBy": sort_by,
            "filterData": [],
        }

        try:
            logger.info(f"POST {url} | search_text='{search_text}'")
            response = requests.post(url, json=payload, timeout=self.timeout)

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Backend API returned status {response.status_code}")
                raise BackendAPIError(
                    f"Backend API returned status {response.status_code}"
                )

        except requests.RequestException as e:
            logger.error(f"Backend API request failed: {e}")
            raise BackendAPIError(
                "The system is currently undergoing maintenance. Please check back later."
            )

    def get_course_url(self, course_id: int) -> str:
        """Build the public URL for a course."""
        return f"{self.base_url}/course/{course_id}"
