"""
Tavily API Service for web search functionality.

This module provides a singleton service for interacting with the Tavily search API.
"""

import os
import logging
from typing import Optional, List, Dict, Any

import httpx

logger = logging.getLogger("rotator_library.web_search")


class TavilyService:
    """
    Tavily API client for web search.

    Provides async search functionality using the Tavily API.
    """

    # Singleton instance
    _instance: Optional["TavilyService"] = None

    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY", "").strip()
        self.search_depth = os.getenv("TAVILY_SEARCH_DEPTH", "basic").lower()
        self.max_results = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
        self.base_url = "https://api.tavily.com"

    @property
    def is_configured(self) -> bool:
        """Check if Tavily is properly configured with an API key."""
        return bool(self.api_key)

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        search_depth: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a Tavily search.

        Args:
            query: The search query
            max_results: Maximum number of results (defaults to configured value)
            search_depth: Search depth - 'basic' or 'advanced' (defaults to configured value)
            include_domains: List of domains to include
            exclude_domains: List of domains to exclude

        Returns:
            Dict containing search results with keys:
            - results: List of search result objects
            - query: The original query
            - response_time: Time taken for the search

        Raises:
            httpx.HTTPError: If the API request fails
        """
        if not self.is_configured:
            return {
                "results": [],
                "query": query,
                "error": "Tavily API key not configured",
            }

        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": search_depth or self.search_depth,
            "max_results": max_results or self.max_results,
            "include_answer": True,
            "include_raw_content": False,
        }

        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/search",
                    json=payload,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"Tavily API error: {e.response.status_code} - {e.response.text}")
                return {
                    "results": [],
                    "query": query,
                    "error": f"Tavily API error: {e.response.status_code}",
                }
            except httpx.RequestError as e:
                logger.error(f"Tavily request error: {e}")
                return {
                    "results": [],
                    "query": query,
                    "error": f"Request error: {str(e)}",
                }


def get_tavily_service() -> TavilyService:
    """
    Get the singleton TavilyService instance.

    Returns:
        The TavilyService singleton instance.
    """
    if TavilyService._instance is None:
        TavilyService._instance = TavilyService()
    return TavilyService._instance
