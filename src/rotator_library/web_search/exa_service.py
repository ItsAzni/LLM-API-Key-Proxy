"""
Exa MCP Service for web search functionality.

This module provides a service for interacting with Exa's free MCP endpoint.
No API key is required - Exa provides this as a public MCP server.
"""

import logging
from typing import Optional, Dict, Any

import httpx

logger = logging.getLogger("rotator_library.web_search")


class ExaService:
    """
    Exa MCP client for web search.

    Uses Exa's free public MCP endpoint at mcp.exa.ai.
    No API key required.
    """

    # Singleton instance
    _instance: Optional["ExaService"] = None

    def __init__(self):
        self.base_url = "https://mcp.exa.ai"
        self.endpoint = "/mcp"
        self.default_num_results = 8
        self.timeout = 25.0  # 25 second timeout like OpenCode uses

    @property
    def is_configured(self) -> bool:
        """Exa MCP is always configured - no API key needed."""
        return True

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        search_type: Optional[str] = None,
        livecrawl: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute an Exa MCP search.

        Args:
            query: The search query
            max_results: Maximum number of results (defaults to 8)
            search_type: Search type - 'auto', 'fast', or 'deep' (defaults to 'auto')
            livecrawl: Live crawl mode - 'fallback' or 'preferred' (defaults to 'fallback')

        Returns:
            Dict containing search results with keys:
            - results: List of search result objects
            - query: The original query
            - answer: Summary if available
            - error: Error message if request failed
        """
        # Build MCP JSON-RPC request
        mcp_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "web_search_exa",
                "arguments": {
                    "query": query,
                    "type": search_type or "auto",
                    "numResults": max_results or self.default_num_results,
                    "livecrawl": livecrawl or "fallback",
                },
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}{self.endpoint}",
                    json=mcp_request,
                    headers={
                        "accept": "application/json, text/event-stream",
                        "content-type": "application/json",
                    },
                )
                response.raise_for_status()

                response_text = response.text

                # Parse SSE response (Exa returns Server-Sent Events)
                results = self._parse_sse_response(response_text, query)
                return results

            except httpx.HTTPStatusError as e:
                logger.error(f"Exa MCP API error: {e.response.status_code} - {e.response.text}")
                return {
                    "results": [],
                    "query": query,
                    "error": f"Exa MCP API error: {e.response.status_code}",
                }
            except httpx.RequestError as e:
                logger.error(f"Exa MCP request error: {e}")
                return {
                    "results": [],
                    "query": query,
                    "error": f"Request error: {str(e)}",
                }
            except Exception as e:
                logger.error(f"Exa MCP unexpected error: {e}")
                return {
                    "results": [],
                    "query": query,
                    "error": f"Unexpected error: {str(e)}",
                }

    def _parse_sse_response(self, response_text: str, query: str) -> Dict[str, Any]:
        """
        Parse SSE (Server-Sent Events) response from Exa MCP.

        The response format is:
        data: {"jsonrpc": "2.0", "result": {"content": [{"type": "text", "text": "..."}]}}

        Args:
            response_text: Raw response text
            query: Original query for error context

        Returns:
            Parsed search results
        """
        import json

        lines = response_text.split("\n")
        for line in lines:
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])  # Remove "data: " prefix
                    if "result" in data and "content" in data["result"]:
                        content = data["result"]["content"]
                        if content and len(content) > 0:
                            # Exa returns results as formatted text
                            text_content = content[0].get("text", "")
                            return self._parse_exa_text_results(text_content, query)
                except json.JSONDecodeError:
                    continue

        return {
            "results": [],
            "query": query,
            "error": "No results found in Exa response",
        }

    def _parse_exa_text_results(self, text: str, query: str) -> Dict[str, Any]:
        """
        Parse Exa's text-formatted results into structured format.

        Exa returns results as formatted text. We'll return it as-is
        since it's already well-formatted for LLM consumption.

        Args:
            text: The text content from Exa
            query: Original query

        Returns:
            Structured results dict
        """
        if not text:
            return {
                "results": [],
                "query": query,
                "error": "Empty response from Exa",
            }

        # Return the text as a single "answer" since Exa already formats it nicely
        return {
            "results": [],  # Exa returns pre-formatted text, not structured results
            "query": query,
            "answer": text,  # Use as answer/summary
        }


def get_exa_service() -> ExaService:
    """
    Get the singleton ExaService instance.

    Returns:
        The ExaService singleton instance.
    """
    if ExaService._instance is None:
        ExaService._instance = ExaService()
    return ExaService._instance
