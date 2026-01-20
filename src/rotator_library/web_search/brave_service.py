"""
Brave Search API Service for web search functionality.

This module provides a singleton service for interacting with the Brave Search API.
Used as a fallback when Tavily is unavailable or exhausted.
"""

import os
import logging
from typing import Optional, Dict, Any

import httpx

logger = logging.getLogger("rotator_library.web_search")


class BraveService:
    """
    Brave Search API client for web search.

    Provides async search functionality using the Brave Search API.
    """

    # Singleton instance
    _instance: Optional["BraveService"] = None

    def __init__(self):
        self.api_key = os.getenv("BRAVE_API_KEY", "").strip()
        self.max_results = int(os.getenv("BRAVE_MAX_RESULTS", "5"))
        self.base_url = "https://api.search.brave.com/res/v1/web/search"

    @property
    def is_configured(self) -> bool:
        """Check if Brave is properly configured with an API key."""
        return bool(self.api_key)

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        country: Optional[str] = None,
        search_lang: Optional[str] = None,
        freshness: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a Brave search.

        Args:
            query: The search query
            max_results: Maximum number of results (defaults to configured value)
            country: Country code for results (e.g., 'US', 'DE')
            search_lang: Language for search (e.g., 'en', 'de')
            freshness: Freshness filter ('pd' = past day, 'pw' = past week, 'pm' = past month)

        Returns:
            Dict containing search results with keys:
            - results: List of search result objects
            - query: The original query
            - answer: Summary if available

        Raises:
            httpx.HTTPError: If the API request fails
        """
        if not self.is_configured:
            return {
                "results": [],
                "query": query,
                "error": "Brave API key not configured",
            }

        params = {
            "q": query,
            "count": max_results or self.max_results,
        }

        if country:
            params["country"] = country
        if search_lang:
            params["search_lang"] = search_lang
        if freshness:
            params["freshness"] = freshness

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(
                    self.base_url,
                    params=params,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()

                # Transform Brave response to match Tavily format
                return self._transform_response(data, query)

            except httpx.HTTPStatusError as e:
                error_msg = f"Brave API error: {e.response.status_code}"
                # Check for quota exhaustion (429 or specific error messages)
                if e.response.status_code == 429:
                    error_msg = "Brave API quota exhausted"
                elif e.response.status_code == 401:
                    error_msg = "Brave API authentication failed"

                logger.error(f"{error_msg} - {e.response.text}")
                return {
                    "results": [],
                    "query": query,
                    "error": error_msg,
                }
            except httpx.RequestError as e:
                logger.error(f"Brave request error: {e}")
                return {
                    "results": [],
                    "query": query,
                    "error": f"Request error: {str(e)}",
                }

    def _transform_response(self, data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Transform Brave API response to match Tavily format.

        This ensures consistent response format regardless of which search provider is used.
        """
        results = []

        # Extract web results
        web_results = data.get("web", {}).get("results", [])
        for item in web_results:
            results.append({
                "title": item.get("title", "Untitled"),
                "url": item.get("url", ""),
                "content": item.get("description", ""),
            })

        # Check for summarizer/answer
        answer = None
        if "summarizer" in data and data["summarizer"].get("key"):
            # Brave has a summarizer but we'd need another call to get it
            # For simplicity, we'll skip this for now
            pass

        return {
            "results": results,
            "query": query,
            "answer": answer,
            "response_time": data.get("query", {}).get("response_time", 0),
        }


def get_brave_service() -> BraveService:
    """
    Get the singleton BraveService instance.

    Returns:
        The BraveService singleton instance.
    """
    if BraveService._instance is None:
        BraveService._instance = BraveService()
    return BraveService._instance
