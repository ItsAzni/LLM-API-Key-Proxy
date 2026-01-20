"""
Web Search Module for LLM-API-Key-Proxy.

Provides web search as a tool that LLMs can call.
Supports multiple providers with automatic fallback:
1. Exa MCP (free, no API key required)
2. Tavily (if configured)
3. Brave Search (if configured)

The proxy injects the web_search tool definition when enabled,
intercepts tool calls, executes searches, and continues the conversation loop.
"""

from .exa_service import ExaService, get_exa_service
from .tavily_service import TavilyService, get_tavily_service
from .brave_service import BraveService, get_brave_service
from .search_service import SearchService, get_search_service
from .tool_handler import (
    WEB_SEARCH_TOOL_DEFINITION,
    inject_web_search_tool,
    has_web_search_tool_call,
    execute_web_search_tool,
    build_tool_result_message,
    detect_freshness_from_query,
)
from .tool_loop import execute_with_tool_loop

__all__ = [
    # Exa (free, primary)
    "ExaService",
    "get_exa_service",
    # Tavily
    "TavilyService",
    "get_tavily_service",
    # Brave
    "BraveService",
    "get_brave_service",
    # Unified search
    "SearchService",
    "get_search_service",
    # Tool handling
    "WEB_SEARCH_TOOL_DEFINITION",
    "inject_web_search_tool",
    "has_web_search_tool_call",
    "execute_web_search_tool",
    "build_tool_result_message",
    "detect_freshness_from_query",
    "execute_with_tool_loop",
]
