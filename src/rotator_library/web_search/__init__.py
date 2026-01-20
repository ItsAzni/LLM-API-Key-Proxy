"""
Web Search Module for LLM-API-Key-Proxy.

Provides Tavily-powered web search as a tool that LLMs can call.
The proxy injects the web_search tool definition when Tavily is configured,
intercepts tool calls, executes searches, and continues the conversation loop.
"""

from .tavily_service import TavilyService, get_tavily_service
from .tool_handler import (
    WEB_SEARCH_TOOL_DEFINITION,
    inject_web_search_tool,
    has_web_search_tool_call,
    execute_web_search_tool,
    build_tool_result_message,
)
from .tool_loop import execute_with_tool_loop

__all__ = [
    "TavilyService",
    "get_tavily_service",
    "WEB_SEARCH_TOOL_DEFINITION",
    "inject_web_search_tool",
    "has_web_search_tool_call",
    "execute_web_search_tool",
    "build_tool_result_message",
    "execute_with_tool_loop",
]
