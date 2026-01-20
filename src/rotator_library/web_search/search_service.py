"""
Unified Search Service with fallback support.

Provides a unified interface for web search that tries providers in order:
1. Exa MCP (free, no API key required)
2. Tavily (if configured)
3. Brave (if configured)
"""

import logging
from typing import Dict, Any, Optional

from .exa_service import get_exa_service
from .tavily_service import get_tavily_service
from .brave_service import get_brave_service

logger = logging.getLogger("rotator_library.web_search")


# Mapping from normalized freshness to Brave API format
BRAVE_FRESHNESS_MAP = {
    "day": "pd",    # past day
    "week": "pw",   # past week
    "month": "pm",  # past month
    "year": "py",   # past year
}


class SearchService:
    """
    Unified search service with automatic fallback.

    Priority order:
    1. Exa MCP (always available - free, no API key)
    2. Tavily (if configured)
    3. Brave (if configured and others fail)
    """

    _instance: Optional["SearchService"] = None

    def __init__(self):
        self.exa = get_exa_service()
        self.tavily = get_tavily_service()
        self.brave = get_brave_service()

    @property
    def is_configured(self) -> bool:
        """Check if at least one search provider is configured."""
        # Exa is always configured (no API key needed)
        return True

    @property
    def configured_providers(self) -> list[str]:
        """Get list of configured provider names."""
        providers = ["exa"]  # Exa is always available
        if self.tavily.is_configured:
            providers.append("tavily")
        if self.brave.is_configured:
            providers.append("brave")
        return providers

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        freshness: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a web search with automatic fallback.

        Tries Exa first (free), then Tavily, then Brave.

        Args:
            query: The search query
            max_results: Maximum number of results
            freshness: Time filter - 'day', 'week', 'month', 'year', or None/any

        Returns:
            Dict containing search results with keys:
            - results: List of search result objects
            - query: The original query
            - answer: Summary if available
            - provider: Which provider returned the results
            - error: Error message if all providers failed
        """
        errors = []

        # Normalize freshness (None or "any" means no filter)
        if freshness == "any":
            freshness = None

        # Try Exa MCP first (free, no API key)
        logger.debug(f"Attempting Exa MCP search for: {query}")
        result = await self.exa.search(
            query,
            max_results=max_results,
            # Exa doesn't support freshness filtering directly
        )

        if "error" not in result or result.get("answer") or result.get("results"):
            result["provider"] = "exa"
            logger.info(f"Exa MCP search successful for: {query}")
            return result

        # Exa failed, record error
        error_msg = result.get("error", "Unknown Exa error")
        errors.append(f"Exa: {error_msg}")
        logger.warning(f"Exa MCP search failed: {error_msg}")

        # Try Tavily as first paid fallback
        if self.tavily.is_configured:
            logger.debug(f"Attempting Tavily search for: {query} (freshness={freshness})")
            result = await self.tavily.search(
                query,
                max_results=max_results,
                time_range=freshness,  # Tavily uses same values: day, week, month, year
            )

            if "error" not in result or result.get("results"):
                result["provider"] = "tavily"
                logger.info(f"Tavily search successful for: {query}")
                return result

            # Tavily failed, record error
            error_msg = result.get("error", "Unknown Tavily error")
            errors.append(f"Tavily: {error_msg}")
            logger.warning(f"Tavily search failed: {error_msg}")

        # Try Brave as last fallback
        if self.brave.is_configured:
            # Map freshness to Brave format
            brave_freshness = BRAVE_FRESHNESS_MAP.get(freshness) if freshness else None

            logger.debug(f"Attempting Brave search for: {query} (freshness={brave_freshness})")
            result = await self.brave.search(
                query,
                max_results=max_results,
                freshness=brave_freshness,
            )

            if "error" not in result or result.get("results"):
                result["provider"] = "brave"
                logger.info(f"Brave search successful for: {query}")
                return result

            # Brave failed, record error
            error_msg = result.get("error", "Unknown Brave error")
            errors.append(f"Brave: {error_msg}")
            logger.warning(f"Brave search failed: {error_msg}")

        # All providers failed
        error_msg = " | ".join(errors)
        logger.error(f"All search providers failed for: {query}")
        return {
            "results": [],
            "query": query,
            "error": error_msg,
            "provider": None,
        }


def get_search_service() -> SearchService:
    """
    Get the singleton SearchService instance.

    Returns:
        The SearchService singleton instance.
    """
    if SearchService._instance is None:
        SearchService._instance = SearchService()
    return SearchService._instance
