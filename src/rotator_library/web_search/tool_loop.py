"""
Tool Loop for agentic web search execution.

Provides a wrapper around the RotatingClient that handles the agentic tool loop:
1. Send request to LLM
2. Check if LLM responds with web_search tool call
3. Execute web search (Tavily or Brave fallback)
4. Add result to messages and continue
5. Repeat until final response or max iterations
"""

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from .search_service import get_search_service
from .tool_handler import (
    extract_all_web_search_tool_calls,
    execute_web_search_tool,
    build_tool_result_message,
    build_assistant_message_with_tool_calls,
)

logger = logging.getLogger("rotator_library.web_search")

# Tool names that should be handled as web search (includes proxy's tool and native provider tools)
WEB_SEARCH_TOOL_NAMES = {
    "web_search",           # Proxy's injected tool
    "web_search_preview",   # OpenAI's native web search
    "gemini3_web_search",   # Gemini 3's native web search
    "google_search",        # Gemini's google search
    "googleSearch",         # Alternative naming
}


async def execute_with_tool_loop(
    client: Any,
    request_data: Dict[str, Any],
    max_tool_iterations: int = 5,
    request: Optional[Any] = None,
    **kwargs,
) -> Union[Any, AsyncGenerator[str, None]]:
    """
    Execute completion with automatic web_search tool handling.

    If model calls web_search tool:
    1. Execute Tavily search
    2. Add tool result to messages
    3. Continue until final response or max iterations

    Args:
        client: The RotatingClient instance
        request_data: The request data dict (will be modified)
        max_tool_iterations: Maximum number of tool call iterations
        request: Optional FastAPI request object for disconnect checks
        **kwargs: Additional arguments passed to acompletion

    Returns:
        For non-streaming: The final response
        For streaming: An async generator that yields SSE events
    """
    is_streaming = request_data.get("stream", False)

    if is_streaming:
        return _streaming_tool_loop(
            client, request_data, max_tool_iterations, request, **kwargs
        )
    else:
        return await _non_streaming_tool_loop(
            client, request_data, max_tool_iterations, request, **kwargs
        )


async def _non_streaming_tool_loop(
    client: Any,
    request_data: Dict[str, Any],
    max_tool_iterations: int,
    request: Optional[Any] = None,
    **kwargs,
) -> Any:
    """
    Handle non-streaming tool loop.

    Iteratively processes tool calls until the model produces a final response.
    """
    search_service = get_search_service()
    messages = list(request_data.get("messages", []))
    iteration = 0
    seen_queries = set()
    early_stop = False

    while iteration < max_tool_iterations:
        # Update messages in request data
        current_request = dict(request_data)
        current_request["messages"] = messages
        current_request["stream"] = False

        # Make the completion call
        response = await client.acompletion(
            request=request, **current_request, **kwargs
        )

        # Check for web_search tool calls
        tool_calls = extract_all_web_search_tool_calls(response)

        if not tool_calls:
            # No web_search tool calls, return the response
            return response

        if not search_service.is_configured:
            # No search provider configured but model tried to use web_search
            logger.warning(
                "Model called web_search but no search provider is configured"
            )
            return response

        logger.info(
            f"Tool loop iteration {iteration + 1}: {len(tool_calls)} web_search call(s)"
        )

        filtered_tool_calls = []
        for query, tool_call_id, freshness in tool_calls:
            key = (query, freshness or "")
            if key in seen_queries:
                continue
            seen_queries.add(key)
            filtered_tool_calls.append((query, tool_call_id, freshness))

        if not filtered_tool_calls:
            logger.warning("Detected repeated web_search calls; stopping tool loop")
            early_stop = True
            break

        # Add assistant message with tool calls to conversation
        assistant_msg = build_assistant_message_with_tool_calls(response)
        messages.append(assistant_msg)

        # Execute all web searches and add results
        for query, tool_call_id, freshness in filtered_tool_calls:
            result = await execute_web_search_tool(query, freshness=freshness)
            tool_result_msg = build_tool_result_message(tool_call_id, result)
            messages.append(tool_result_msg)

        iteration += 1

    if not early_stop:
        logger.warning(f"Tool loop reached max iterations ({max_tool_iterations})")
    # Make one final call without allowing further tool calls
    final_request = dict(request_data)
    final_request["messages"] = messages
    final_request["stream"] = False
    # Remove tools to prevent further tool calls
    final_request.pop("tools", None)
    final_request.pop("tool_choice", None)

    return await client.acompletion(request=request, **final_request, **kwargs)


async def _streaming_tool_loop(
    client: Any,
    request_data: Dict[str, Any],
    max_tool_iterations: int,
    request: Optional[Any] = None,
    **kwargs,
) -> AsyncGenerator[str, None]:
    """
    Handle streaming tool loop with real-time streaming.

    Streams chunks to client in real-time while accumulating to detect tool calls.
    If tool calls are detected, executes them and streams another response.
    """
    search_service = get_search_service()
    messages = list(request_data.get("messages", []))
    iteration = 0
    seen_queries = set()
    early_stop = False

    while iteration < max_tool_iterations:
        # Update messages in request data
        current_request = dict(request_data)
        current_request["messages"] = messages
        current_request["stream"] = True

        accumulated_response = _create_empty_accumulated_response()

        stream = await client.acompletion(request=request, **current_request, **kwargs)

        done_chunk = None
        last_chunk_with_finish_reason = None  # Track last chunk that had finish_reason

        # Stream chunks in real-time while accumulating for tool detection
        async for chunk in stream:
            if chunk.strip() == "data: [DONE]":
                done_chunk = chunk
                continue

            # Check if this chunk has finish_reason before we strip it
            if chunk.startswith("data: "):
                try:
                    data_str = chunk[6:].strip()
                    if data_str != "[DONE]":
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices and choices[0].get("finish_reason"):
                            last_chunk_with_finish_reason = chunk
                except json.JSONDecodeError:
                    pass

            sanitized_chunk = _sanitize_stream_chunk(chunk)
            yield sanitized_chunk  # Stream immediately to client!
            _accumulate_chunk(accumulated_response, chunk)

        # After stream ends, check for tool calls
        web_search_calls, has_other_tools = _extract_tool_calls_from_accumulated(accumulated_response)

        # Debug: log what tools were accumulated
        if accumulated_response["tool_calls"]:
            tool_names = [tc.get("function", {}).get("name", "?") for tc in accumulated_response["tool_calls"]]
            logger.debug(f"Accumulated tool calls: {tool_names}, web_search_calls={len(web_search_calls)}, has_other_tools={has_other_tools}")

        # If there are non-web_search tool calls, the client needs to handle them
        # Re-emit the final chunk with finish_reason so client knows to execute tools
        if has_other_tools and not web_search_calls:
            # Only other tools, no web_search - client handles everything
            if last_chunk_with_finish_reason:
                yield last_chunk_with_finish_reason
            if done_chunk:
                yield done_chunk
            return

        if not web_search_calls:
            # No tool calls - we're done (chunks already streamed)
            if done_chunk:
                yield done_chunk
            return

        if not search_service.is_configured:
            logger.warning(
                "Model called web_search but no search provider is configured"
            )
            return

        logger.info(
            f"Streaming tool loop iteration {iteration + 1}: {len(web_search_calls)} web_search call(s)"
        )

        filtered_tool_calls = []
        for query, tool_call_id, freshness in web_search_calls:
            key = (query, freshness or "")
            if key in seen_queries:
                continue
            seen_queries.add(key)
            filtered_tool_calls.append((query, tool_call_id, freshness))

        if not filtered_tool_calls:
            logger.warning("Detected repeated web_search calls; stopping tool loop")
            early_stop = True
            break

        # Build assistant message from accumulated response
        assistant_msg = _build_assistant_message_from_accumulated(accumulated_response)
        messages.append(assistant_msg)

        # Execute all web searches and add results
        for query, tool_call_id, freshness in filtered_tool_calls:
            result = await execute_web_search_tool(query, freshness=freshness)
            tool_result_msg = build_tool_result_message(tool_call_id, result)
            messages.append(tool_result_msg)

        iteration += 1
        # Loop continues - will make another streaming request with tool results

    # Max iterations reached or early stop - make final call without tools
    if not early_stop:
        logger.warning(
            f"Streaming tool loop reached max iterations ({max_tool_iterations})"
        )
    final_request = dict(request_data)
    final_request["messages"] = messages
    final_request["stream"] = True
    final_request.pop("tools", None)
    final_request.pop("tool_choice", None)

    stream = await client.acompletion(request=request, **final_request, **kwargs)
    async for chunk in stream:
        yield chunk


def _create_empty_accumulated_response() -> Dict[str, Any]:
    """Create an empty structure for accumulating streaming response."""
    return {
        "content": "",
        "tool_calls": [],
        "tool_call_id_map": {},
        "next_tool_call_index": 0,
        "finish_reason": None,
    }


def _accumulate_chunk(accumulated: Dict[str, Any], chunk: str) -> None:
    """
    Accumulate a streaming chunk into the response structure.

    Parses SSE format and extracts content/tool_calls.
    """
    if not chunk.startswith("data: "):
        return

    data_str = chunk[6:].strip()  # Remove "data: " prefix
    if data_str == "[DONE]":
        return

    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        return

    choices = data.get("choices", [])
    if not choices:
        return

    delta = choices[0].get("delta", {})
    finish_reason = choices[0].get("finish_reason")

    # Accumulate content
    if delta.get("content"):
        accumulated["content"] += delta["content"]

    # Accumulate tool calls
    if delta.get("tool_calls"):
        for tc in delta["tool_calls"]:
            index = tc.get("index")
            call_id = tc.get("id")
            if index is None:
                if call_id in accumulated["tool_call_id_map"]:
                    index = accumulated["tool_call_id_map"][call_id]
                else:
                    index = accumulated["next_tool_call_index"]
                    accumulated["next_tool_call_index"] += 1
                    if call_id:
                        accumulated["tool_call_id_map"][call_id] = index
            elif call_id and call_id not in accumulated["tool_call_id_map"]:
                accumulated["tool_call_id_map"][call_id] = index
            # Extend list if needed
            while len(accumulated["tool_calls"]) <= index:
                accumulated["tool_calls"].append(
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                )

            tool_call = accumulated["tool_calls"][index]

            if tc.get("id"):
                tool_call["id"] = tc["id"]
            if tc.get("type"):
                tool_call["type"] = tc["type"]
            if tc.get("function"):
                func = tc["function"]
                if func.get("name"):
                    tool_call["function"]["name"] = func["name"]
                if func.get("arguments"):
                    tool_call["function"]["arguments"] += func["arguments"]

    if finish_reason:
        accumulated["finish_reason"] = finish_reason


def _sanitize_stream_chunk(chunk):
    """Remove web_search tool calls and tool_calls finish_reason from streamed chunks."""
    if not chunk.startswith("data: "):
        return chunk

    data_str = chunk[6:].strip()
    if data_str == "[DONE]":
        return chunk

    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        return chunk

    choices = data.get("choices", [])
    if not choices:
        return chunk

    changed = False
    choice = choices[0]
    delta = choice.get("delta", {})
    tool_calls = delta.get("tool_calls")
    if tool_calls:
        filtered = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                filtered.append(tc)
                continue
            func = tc.get("function") or {}
            name = func.get("name") if isinstance(func, dict) else None
            if name not in WEB_SEARCH_TOOL_NAMES:
                filtered.append(tc)

        if len(filtered) != len(tool_calls):
            changed = True
            if filtered:
                delta["tool_calls"] = filtered
            else:
                delta.pop("tool_calls", None)

    if choice.get("finish_reason") == "tool_calls":
        choice["finish_reason"] = None
        changed = True

    if not changed:
        return chunk

    return f"data: {json.dumps(data)}\n\n"


def _extract_tool_calls_from_accumulated(
    accumulated: Dict[str, Any],
) -> tuple[List[tuple[str, str, Optional[str]]], bool]:
    """
    Extract web_search tool calls from accumulated response.

    Recognizes both the proxy's injected 'web_search' tool and native
    Gemini search tools (gemini3_web_search, google_search, etc.)

    Returns:
        Tuple of (web_search_calls, has_other_tools) where:
        - web_search_calls: list of (query, tool_call_id, freshness) tuples
        - has_other_tools: True if there are non-web_search tool calls
    """
    web_search_calls = []
    has_other_tools = False

    for tool_call in accumulated["tool_calls"]:
        if tool_call.get("type") == "function":
            func = tool_call.get("function", {})
            func_name = func.get("name", "")
            if func_name in WEB_SEARCH_TOOL_NAMES:
                try:
                    args = json.loads(func.get("arguments", "{}"))
                    # Handle different argument formats:
                    # - web_search uses "query"
                    # - gemini3_web_search might use "query" or be in a different format
                    query = args.get("query", "")
                    if not query:
                        # Try alternative field names
                        query = args.get("search_query", "") or args.get("q", "")
                    tool_call_id = tool_call.get("id", "")
                    freshness = args.get("freshness")  # May be None
                    if query and tool_call_id:
                        web_search_calls.append((query, tool_call_id, freshness))
                    elif tool_call_id:
                        # Even without a query, track it as a search tool call
                        # so we don't treat it as "other tools"
                        logger.warning(
                            f"Search tool '{func_name}' called without query, skipping"
                        )
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to parse tool call arguments: {func.get('arguments')}"
                    )
            elif func_name:  # Has a name but not a search tool
                has_other_tools = True

    return web_search_calls, has_other_tools


def _build_assistant_message_from_accumulated(
    accumulated: Dict[str, Any],
) -> Dict[str, Any]:
    """Build an assistant message from accumulated streaming response."""
    message = {
        "role": "assistant",
        "content": accumulated["content"] or None,
    }

    if accumulated["tool_calls"]:
        message["tool_calls"] = accumulated["tool_calls"]

    return message
