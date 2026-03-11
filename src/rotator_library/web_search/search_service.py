"""
Unified Search Service with fallback support.

Provides a unified interface for web search that tries providers in order.
Default priority: SearXNG → Exa → Tavily → Brave
Configurable via WEB_SEARCH_PRIORITY environment variable.
"""

import logging
import os
from typing import Dict, Any, Optional, List

from .exa_service import get_exa_service
from .tavily_service import get_tavily_service
from .brave_service import get_brave_service
from .searxng_service import get_searxng_service

logger = logging.getLogger("rotator_library.web_search")


# Valid provider names
VALID_PROVIDERS = {"searxng", "exa", "tavily", "brave"}

# Default priority order
DEFAULT_PRIORITY = ["searxng", "exa", "tavily", "brave"]


def get_search_priority() -> List[str]:
    """
    Get the configured search provider priority order.

    Reads from WEB_SEARCH_PRIORITY environment variable.
    Format: comma-separated list of provider names (e.g., "tavily,exa,brave")

    Returns:
        List of provider names in priority order
    """
    priority_str = os.environ.get("WEB_SEARCH_PRIORITY", "").strip()

    if not priority_str:
        return DEFAULT_PRIORITY.copy()

    # Parse the priority string
    priority = []
    for provider in priority_str.split(","):
        provider = provider.strip().lower()
        if provider in VALID_PROVIDERS:
            if provider not in priority:  # Avoid duplicates
                priority.append(provider)
        elif provider:
            logger.warning(
                f"Unknown web search provider '{provider}' in WEB_SEARCH_PRIORITY. "
                f"Valid providers: {', '.join(sorted(VALID_PROVIDERS))}"
            )

    if not priority:
        logger.warning(
            f"WEB_SEARCH_PRIORITY contains no valid providers. Using default: {DEFAULT_PRIORITY}"
        )
        return DEFAULT_PRIORITY.copy()

    # Add any missing providers at the end (so they're still available as fallbacks)
    for provider in DEFAULT_PRIORITY:
        if provider not in priority:
            priority.append(provider)

    return priority


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

    Priority order is configurable via WEB_SEARCH_PRIORITY environment variable.
    Default: SearXNG → Exa → Tavily → Brave

    Example: WEB_SEARCH_PRIORITY=tavily,searxng,brave,exa
    """

    _instance: Optional["SearchService"] = None

    def __init__(self):
        self.searxng = get_searxng_service()
        self.exa = get_exa_service()
        self.tavily = get_tavily_service()
        self.brave = get_brave_service()

        # Map provider names to service instances
        self._providers = {
            "searxng": self.searxng,
            "exa": self.exa,
            "tavily": self.tavily,
            "brave": self.brave,
        }

        # Get priority order from config
        self._priority = get_search_priority()
        logger.info(f"Web search provider priority: {' → '.join(self._priority)}")

    @property
    def is_configured(self) -> bool:
        """Check if at least one search provider is configured."""
        # Exa is always configured (no API key needed)
        return True

    @property
    def configured_providers(self) -> list[str]:
        """Get list of configured provider names in priority order."""
        providers = []
        for name in self._priority:
            if name == "searxng" and self.searxng.is_configured:
                providers.append("searxng")
            elif name == "exa":
                providers.append("exa")  # Exa is always available
            elif name == "tavily" and self.tavily.is_configured:
                providers.append("tavily")
            elif name == "brave" and self.brave.is_configured:
                providers.append("brave")
        return providers

    async def _search_with_provider(
        self,
        provider_name: str,
        query: str,
        max_results: Optional[int],
        freshness: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt search with a specific provider.

        Returns result dict on success, None on failure.
        """
        if provider_name == "searxng":
            if not self.searxng.is_configured:
                return None

            logger.debug(f"Attempting SearXNG search for: {query} (freshness={freshness})")
            result = await self.searxng.search(
                query,
                max_results=max_results,
                freshness=freshness,
            )

            if "error" not in result or result.get("results"):
                result["provider"] = "searxng"
                logger.info(f"SearXNG search successful for: {query}")
                return result
            return None

        elif provider_name == "exa":
            logger.debug(f"Attempting Exa MCP search for: {query}")
            result = await self.exa.search(query, max_results=max_results)

            if "error" not in result or result.get("answer") or result.get("results"):
                result["provider"] = "exa"
                logger.info(f"Exa MCP search successful for: {query}")
                return result
            return None

        elif provider_name == "tavily":
            if not self.tavily.is_configured:
                return None

            logger.debug(f"Attempting Tavily search for: {query} (freshness={freshness})")
            result = await self.tavily.search(
                query,
                max_results=max_results,
                time_range=freshness,
            )

            if "error" not in result or result.get("results"):
                result["provider"] = "tavily"
                logger.info(f"Tavily search successful for: {query}")
                return result
            return None

        elif provider_name == "brave":
            if not self.brave.is_configured:
                return None

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
            return None

        return None

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        freshness: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a web search with automatic fallback.

        Tries providers in configured priority order.

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

        # Try providers in priority order
        for provider_name in self._priority:
            result = await self._search_with_provider(
                provider_name, query, max_results, freshness
            )

            if result is not None:
                return result

            # Record error for this provider
            provider_service = self._providers.get(provider_name)
            if provider_service:
                # Check if provider was even configured
                if provider_name == "exa" or (
                    hasattr(provider_service, "is_configured")
                    and provider_service.is_configured
                ):
                    errors.append(f"{provider_name.capitalize()}: search failed")
                    logger.warning(f"{provider_name.capitalize()} search failed for: {query}")

        # All providers failed
        error_msg = " | ".join(errors) if errors else "No configured providers available"
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
