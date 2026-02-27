"""
SearXNG Search Service for web search functionality.

This module provides a singleton service for interacting with a self-hosted
SearXNG instance. Used as the default/highest-priority search provider.
"""

import os
import logging
from typing import Optional, Dict, Any

import httpx

logger = logging.getLogger("rotator_library.web_search")

# Map proxy freshness values to SearXNG time_range values.
# SearXNG supports: day, month, year (no week).
SEARXNG_FRESHNESS_MAP = {
    "day": "day",
    "week": "month",   # closest available option
    "month": "month",
    "year": "year",
}


class SearXNGService:
    """
    SearXNG search client for web search.

    Provides async search functionality using a self-hosted SearXNG instance.
    """

    # Singleton instance
    _instance: Optional["SearXNGService"] = None

    def __init__(self):
        self.base_url = os.getenv("SEARXNG_URL", "").strip().rstrip("/")
        self.max_results = int(os.getenv("SEARXNG_MAX_RESULTS", "5"))

    @property
    def is_configured(self) -> bool:
        """Check if SearXNG is properly configured with a base URL."""
        return bool(self.base_url)

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        freshness: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a SearXNG search.

        Args:
            query: The search query
            max_results: Maximum number of results (defaults to configured value)
            freshness: Freshness filter ('day', 'week', 'month', 'year')

        Returns:
            Dict containing search results with keys:
            - results: List of search result objects
            - query: The original query
            - answer: Summary if available
            - error: Error message if search failed
        """
        if not self.is_configured:
            return {
                "results": [],
                "query": query,
                "error": "SearXNG URL not configured",
            }

        params = {
            "q": query,
            "format": "json",
        }

        time_range = SEARXNG_FRESHNESS_MAP.get(freshness) if freshness else None
        if time_range:
            params["time_range"] = time_range

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(
                    f"{self.base_url}/search",
                    params=params,
                )
                response.raise_for_status()
                data = response.json()

                return self._transform_response(
                    data, query, max_results or self.max_results
                )

            except httpx.HTTPStatusError as e:
                error_msg = f"SearXNG API error: {e.response.status_code}"
                logger.error(f"{error_msg} - {e.response.text}")
                return {
                    "results": [],
                    "query": query,
                    "error": error_msg,
                }
            except httpx.RequestError as e:
                logger.error(f"SearXNG request error: {e}")
                return {
                    "results": [],
                    "query": query,
                    "error": f"Request error: {str(e)}",
                }

    def _transform_response(
        self, data: Dict[str, Any], query: str, max_results: int
    ) -> Dict[str, Any]:
        """
        Transform SearXNG JSON response to the common search result format.
        """
        results = []

        for item in data.get("results", []):
            if len(results) >= max_results:
                break
            results.append({
                "title": item.get("title", "Untitled"),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            })

        # Join answers if present
        answers = data.get("answers", [])
        answer = " ".join(answers) if answers else None

        return {
            "results": results,
            "query": query,
            "answer": answer,
        }


def get_searxng_service() -> SearXNGService:
    """
    Get the singleton SearXNGService instance.

    Returns:
        The SearXNGService singleton instance.
    """
    if SearXNGService._instance is None:
        SearXNGService._instance = SearXNGService()
    return SearXNGService._instance
