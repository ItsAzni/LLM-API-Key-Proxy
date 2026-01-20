"""
Tool Handler for web search functionality.

Provides the web_search tool definition and handlers for injecting tools
into requests and processing tool calls from model responses.
"""

import json
import logging
from typing import Dict, Any, Optional, Tuple, List

from .tavily_service import get_tavily_service

logger = logging.getLogger("rotator_library.web_search")


# Web search tool definition in OpenAI function format (universal fallback)
WEB_SEARCH_TOOL_DEFINITION: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information. Use for questions about "
            "recent events, weather, news, prices, or anything requiring up-to-date data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                }
            },
            "required": ["query"],
        },
    },
}


def inject_web_search_tool(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add web_search tool to request if Tavily is configured and tool not already present.

    Args:
        request_data: The original request data dict

    Returns:
        Modified request data with web_search tool injected (if applicable)
    """
    tavily_service = get_tavily_service()
    if not tavily_service.is_configured:
        return request_data

    # Check if tools already exist in request
    tools = request_data.get("tools", [])
    if tools is None:
        tools = []

    # Check if web_search tool is already present
    for tool in tools:
        if tool.get("type") == "function":
            func = tool.get("function", {})
            if func.get("name") == "web_search":
                # Already present, don't inject
                return request_data
        # Also check for OpenAI Responses API format
        if tool.get("type") in ("web_search", "web_search_preview"):
            # Native web search requested, don't inject our version
            return request_data
        # Check for Anthropic format
        if tool.get("type") == "web_search_20250305":
            return request_data

    # Inject our web_search tool
    tools = list(tools)  # Make a copy
    tools.append(WEB_SEARCH_TOOL_DEFINITION)

    # Return modified request data
    result = dict(request_data)
    result["tools"] = tools
    return result


def has_web_search_tool_call(response: Any) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if response contains a web_search tool call.

    Args:
        response: The model response (can be dict or object)

    Returns:
        Tuple of (has_tool_call, query, tool_call_id)
    """
    # Handle both dict and object responses
    if hasattr(response, "model_dump"):
        response_dict = response.model_dump()
    elif isinstance(response, dict):
        response_dict = response
    else:
        return False, None, None

    # Check choices for tool calls
    choices = response_dict.get("choices", [])
    for choice in choices:
        message = choice.get("message", {})
        tool_calls = message.get("tool_calls", [])

        for tool_call in tool_calls:
            if tool_call.get("type") == "function":
                function = tool_call.get("function", {})
                if function.get("name") == "web_search":
                    # Extract query from arguments
                    args_str = function.get("arguments", "{}")
                    try:
                        args = json.loads(args_str)
                        query = args.get("query")
                        tool_call_id = tool_call.get("id")
                        return True, query, tool_call_id
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool call arguments: {args_str}")
                        return True, None, tool_call.get("id")

    return False, None, None


def extract_all_web_search_tool_calls(response: Any) -> List[Tuple[str, str]]:
    """
    Extract all web_search tool calls from a response.

    Args:
        response: The model response (can be dict or object)

    Returns:
        List of tuples (query, tool_call_id)
    """
    result = []

    # Handle both dict and object responses
    if hasattr(response, "model_dump"):
        response_dict = response.model_dump()
    elif isinstance(response, dict):
        response_dict = response
    else:
        return result

    # Check choices for tool calls
    choices = response_dict.get("choices", [])
    for choice in choices:
        message = choice.get("message", {})
        tool_calls = message.get("tool_calls", [])

        for tool_call in tool_calls:
            if tool_call.get("type") == "function":
                function = tool_call.get("function", {})
                if function.get("name") == "web_search":
                    args_str = function.get("arguments", "{}")
                    try:
                        args = json.loads(args_str)
                        query = args.get("query", "")
                        tool_call_id = tool_call.get("id", "")
                        if query and tool_call_id:
                            result.append((query, tool_call_id))
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool call arguments: {args_str}")

    return result


async def execute_web_search_tool(query: str) -> str:
    """
    Execute Tavily search and return formatted result string.

    Args:
        query: The search query

    Returns:
        Formatted string containing search results
    """
    if not query:
        return "[Error: No search query provided]"

    tavily_service = get_tavily_service()
    if not tavily_service.is_configured:
        return "[Error: Web search is not configured]"

    logger.info(f"Executing web search: {query}")

    try:
        result = await tavily_service.search(query)

        if "error" in result:
            return f"[Search Error: {result['error']}]"

        # Format results
        output_parts = ["[Search Results]", ""]

        # Include Tavily's answer if available
        if result.get("answer"):
            output_parts.append(f"Summary: {result['answer']}")
            output_parts.append("")

        # Format individual results
        results = result.get("results", [])
        if not results:
            return "[No search results found]"

        for i, item in enumerate(results, 1):
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            content = item.get("content", "")

            output_parts.append(f"{i}. {title}")
            if url:
                output_parts.append(f"   URL: {url}")
            if content:
                # Truncate content if too long
                if len(content) > 500:
                    content = content[:500] + "..."
                output_parts.append(f"   {content}")
            output_parts.append("")

        return "\n".join(output_parts)

    except Exception as e:
        logger.error(f"Web search execution error: {e}")
        return f"[Search Error: {str(e)}]"


def build_tool_result_message(tool_call_id: str, result: str) -> Dict[str, Any]:
    """
    Build tool result message for conversation continuation.

    Args:
        tool_call_id: The ID of the tool call being responded to
        result: The search result content

    Returns:
        Dict representing the tool result message
    """
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": result,
    }


def build_assistant_message_with_tool_calls(response: Any) -> Dict[str, Any]:
    """
    Build an assistant message from a response that contains tool calls.

    This is used to add the assistant's tool call to the conversation history
    before adding the tool result.

    Args:
        response: The model response containing tool calls

    Returns:
        Dict representing the assistant message with tool calls
    """
    if hasattr(response, "model_dump"):
        response_dict = response.model_dump()
    elif isinstance(response, dict):
        response_dict = response
    else:
        return {"role": "assistant", "content": "", "tool_calls": []}

    choices = response_dict.get("choices", [])
    if not choices:
        return {"role": "assistant", "content": "", "tool_calls": []}

    message = choices[0].get("message", {})
    return {
        "role": "assistant",
        "content": message.get("content") or "",
        "tool_calls": message.get("tool_calls", []),
    }
