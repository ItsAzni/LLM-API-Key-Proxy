"""
Format translation functions between Ollama and OpenAI API formats.

This module provides functions to convert requests and responses between
Ollama's native API format and OpenAI's Chat Completions API format.
This enables Raycast AI and other Ollama clients to use the proxy.
"""

import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    OllamaChatRequest,
    OllamaChatMessage,
    OllamaChunkResponse,
    OllamaMessageContent,
    OllamaToolCall,
    OllamaToolCallFunction,
    OllamaModelInfo,
    OllamaModelDetails,
    OllamaShowResponse,
)


def generate_model_display_name(model_id: str) -> str:
    """
    Generate a human-readable display name from a model ID.

    Args:
        model_id: Internal model ID (e.g., "antigravity/claude-sonnet-4")

    Returns:
        Display name (e.g., "Claude Sonnet 4")

    Examples:
        antigravity/claude-sonnet-4 -> Claude Sonnet 4
        gemini/gemini-2.5-flash -> Gemini 2.5 Flash
        openai/gpt-4o -> GPT 4o
        anthropic/claude-3-opus -> Claude 3 Opus
    """
    # Strip provider prefix
    if "/" in model_id:
        name = model_id.split("/")[-1]
    else:
        name = model_id

    # Replace separators with spaces
    name = name.replace("-", " ").replace("_", " ")

    # Title case each word
    words = name.split()
    result = []
    for word in words:
        # Handle version numbers (keep as-is)
        if re.match(r"^\d", word):
            result.append(word)
        # Handle special acronyms
        elif word.lower() in ("gpt", "llm", "ai"):
            result.append(word.upper())
        else:
            result.append(word.capitalize())

    return " ".join(result)


def model_display_name_to_id(
    display_name: str, model_registry: Dict[str, str]
) -> Optional[str]:
    """
    Look up the internal model ID from a display name.

    Args:
        display_name: Display name shown to user (e.g., "Claude Sonnet 4")
        model_registry: Dict mapping display names to model IDs

    Returns:
        Internal model ID or None if not found
    """
    # Exact match first
    if display_name in model_registry:
        return model_registry[display_name]

    # Case-insensitive match
    lower_name = display_name.lower()
    for name, model_id in model_registry.items():
        if name.lower() == lower_name:
            return model_id

    # Try matching by model ID itself (in case user sends the ID)
    for name, model_id in model_registry.items():
        if model_id == display_name or model_id.lower() == lower_name:
            return model_id

    return None


def ollama_to_openai_messages(messages: List[OllamaChatMessage]) -> List[Dict[str, Any]]:
    """
    Convert Ollama message format to OpenAI format.

    Key differences:
    - Ollama: images in separate array as base64 strings
    - OpenAI: images as data URIs in content array

    Args:
        messages: List of OllamaChatMessage objects

    Returns:
        List of messages in OpenAI format
    """
    openai_messages = []
    # Track tool call IDs for matching tool responses
    pending_tool_call_ids: List[str] = []

    for msg in messages:
        role = msg.role
        content = msg.content or ""

        # Handle tool responses (need to match with tool call IDs)
        if role == "tool":
            # Use the next pending tool call ID if available
            tool_call_id = (
                pending_tool_call_ids.pop(0)
                if pending_tool_call_ids
                else f"call_{uuid.uuid4().hex[:9]}"
            )
            openai_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content,
                }
            )
            continue

        # Handle images (Ollama uses separate images array)
        if msg.images and role == "user":
            openai_content = [{"type": "text", "text": content}]
            for img_base64 in msg.images:
                # Ollama sends raw base64, need to add data URI prefix
                # Try to detect image type or default to jpeg
                if img_base64.startswith("/9j/"):
                    media_type = "image/jpeg"
                elif img_base64.startswith("iVBOR"):
                    media_type = "image/png"
                elif img_base64.startswith("R0lGO"):
                    media_type = "image/gif"
                elif img_base64.startswith("UklGR"):
                    media_type = "image/webp"
                else:
                    media_type = "image/jpeg"  # Default

                openai_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{img_base64}"},
                    }
                )
            openai_messages.append({"role": role, "content": openai_content})
            continue

        # Handle assistant messages with tool calls
        if role == "assistant" and msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                call_id = f"call_{uuid.uuid4().hex[:9]}"
                pending_tool_call_ids.append(call_id)
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            # Ollama sends arguments as object, OpenAI needs JSON string
                            "arguments": json.dumps(tc.function.arguments),
                        },
                    }
                )
            openai_messages.append(
                {
                    "role": "assistant",
                    "content": content if content else None,
                    "tool_calls": tool_calls,
                }
            )
            continue

        # Regular message
        openai_messages.append({"role": role, "content": content})

    return openai_messages


def ollama_to_openai_tools(
    ollama_tools: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """
    Convert Ollama/Raycast tool definitions to OpenAI format.

    Raycast sends tools with type="local_tool" and function definition.

    Args:
        ollama_tools: List of tools in Ollama/Raycast format

    Returns:
        List of tools in OpenAI format, or None if no tools
    """
    if not ollama_tools:
        return None

    openai_tools = []
    for tool in ollama_tools:
        tool_type = tool.get("type", "function")

        # Handle Raycast's local_tool format
        if tool_type == "local_tool":
            func = tool.get("function", {})
            params = func.get("parameters", {})
            # Ensure parameters has proper schema structure
            if not params:
                params = {"type": "object", "properties": {}, "required": []}
            elif "type" not in params:
                params = {
                    "type": "object",
                    "properties": params,
                    "required": [],
                }

            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "parameters": params,
                    },
                }
            )
        elif tool_type == "function":
            # Already in OpenAI-like format
            openai_tools.append(tool)

    return openai_tools if openai_tools else None


def ollama_to_openai_request(
    request: OllamaChatRequest, model_id: str
) -> Dict[str, Any]:
    """
    Translate a complete Ollama chat request to OpenAI format.

    Args:
        request: OllamaChatRequest object
        model_id: Internal model ID (provider/model format)

    Returns:
        Dictionary containing OpenAI-compatible request parameters
    """
    openai_messages = ollama_to_openai_messages(request.messages)
    openai_tools = ollama_to_openai_tools(
        [t.model_dump() for t in request.tools] if request.tools else None
    )

    openai_request = {
        "model": model_id,
        "messages": openai_messages,
        "stream": request.stream,
    }

    if openai_tools:
        openai_request["tools"] = openai_tools

    # Handle options (temperature, top_p, etc.)
    if request.options:
        if "temperature" in request.options:
            openai_request["temperature"] = request.options["temperature"]
        if "top_p" in request.options:
            openai_request["top_p"] = request.options["top_p"]
        if "top_k" in request.options:
            openai_request["top_k"] = request.options["top_k"]
        if "num_predict" in request.options:
            openai_request["max_tokens"] = request.options["num_predict"]
        if "stop" in request.options:
            openai_request["stop"] = request.options["stop"]

    return openai_request


def openai_to_ollama_chunk(
    chunk_data: Dict[str, Any],
    model_name: str,
    accumulated_tool_calls: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Tuple[OllamaChunkResponse, Optional[Dict[int, Dict[str, Any]]]]:
    """
    Convert an OpenAI streaming chunk to Ollama format.

    Args:
        chunk_data: OpenAI chunk dictionary
        model_name: Display name for the model
        accumulated_tool_calls: Dict to accumulate tool call fragments

    Returns:
        Tuple of (OllamaChunkResponse, updated accumulated_tool_calls)
    """
    if accumulated_tool_calls is None:
        accumulated_tool_calls = {}

    choice = chunk_data.get("choices", [{}])[0] if chunk_data.get("choices") else {}
    delta = choice.get("delta", {})
    finish_reason = choice.get("finish_reason")

    # Extract content
    content = delta.get("content", "")

    # Extract thinking/reasoning content
    thinking = None
    reasoning = delta.get("reasoning") or delta.get("reasoning_content")
    if reasoning:
        thinking = reasoning

    # Handle Google Gemini's extra_content format for thinking
    extra_content = delta.get("extra_content", {})
    google_thought = extra_content.get("google", {}).get("thought")
    if google_thought:
        thinking = content.replace("<thought>", "").replace("</thought>", "")
        content = ""

    # Accumulate tool calls
    tool_calls_delta = delta.get("tool_calls", [])
    for tc_chunk in tool_calls_delta:
        index = tc_chunk.get("index", 0)
        if index not in accumulated_tool_calls:
            accumulated_tool_calls[index] = {
                "id": tc_chunk.get("id", ""),
                "function": {"name": "", "arguments": ""},
            }

        if tc_chunk.get("id"):
            accumulated_tool_calls[index]["id"] = tc_chunk["id"]
        if tc_chunk.get("function"):
            func = tc_chunk["function"]
            if func.get("name"):
                accumulated_tool_calls[index]["function"]["name"] += func["name"]
            if func.get("arguments"):
                accumulated_tool_calls[index]["function"]["arguments"] += func[
                    "arguments"
                ]

    # Build tool calls for final chunk
    final_tool_calls = None
    done_reason = None

    if finish_reason:
        done_reason = "stop"
        if finish_reason == "tool_calls":
            done_reason = "tool_calls"
        elif finish_reason == "length":
            done_reason = "length"

        # Convert accumulated tool calls to Ollama format
        if accumulated_tool_calls:
            final_tool_calls = []
            for idx in sorted(accumulated_tool_calls.keys()):
                tc = accumulated_tool_calls[idx]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}

                final_tool_calls.append(
                    OllamaToolCall(
                        function=OllamaToolCallFunction(
                            name=tc["function"]["name"], arguments=args
                        )
                    )
                )

    message = OllamaMessageContent(
        role="assistant",
        content=content,
        thinking=thinking,
        tool_calls=final_tool_calls,
    )

    response = OllamaChunkResponse(
        model=model_name,
        created_at=datetime.utcnow().isoformat() + "Z",
        message=message,
        done=finish_reason is not None,
        done_reason=done_reason,
    )

    return response, accumulated_tool_calls


def openai_to_ollama_response(
    openai_response: Dict[str, Any], model_name: str
) -> OllamaChunkResponse:
    """
    Convert an OpenAI non-streaming response to Ollama format.

    Args:
        openai_response: OpenAI response dictionary
        model_name: Display name for the model

    Returns:
        OllamaChunkResponse (single chunk with done=True)
    """
    choice = openai_response.get("choices", [{}])[0]
    message = choice.get("message", {})
    finish_reason = choice.get("finish_reason", "stop")

    content = message.get("content", "")

    # Handle thinking/reasoning
    thinking = message.get("reasoning_content") or message.get("reasoning")

    # Handle tool calls
    tool_calls = None
    openai_tool_calls = message.get("tool_calls", [])
    if openai_tool_calls:
        tool_calls = []
        for tc in openai_tool_calls:
            func = tc.get("function", {})
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                OllamaToolCall(
                    function=OllamaToolCallFunction(name=func.get("name", ""), arguments=args)
                )
            )

    done_reason = "stop"
    if finish_reason == "tool_calls":
        done_reason = "tool_calls"
    elif finish_reason == "length":
        done_reason = "length"

    # Add usage stats if available
    usage = openai_response.get("usage", {})

    response = OllamaChunkResponse(
        model=model_name,
        created_at=datetime.utcnow().isoformat() + "Z",
        message=OllamaMessageContent(
            role="assistant",
            content=content or "",
            thinking=thinking,
            tool_calls=tool_calls,
        ),
        done=True,
        done_reason=done_reason,
        prompt_eval_count=usage.get("prompt_tokens"),
        eval_count=usage.get("completion_tokens"),
    )

    return response


def generate_ollama_model_info(
    model_id: str,
    context_length: int = 128000,
    capabilities: Optional[List[str]] = None,
) -> OllamaModelInfo:
    """
    Generate Ollama model info from internal model ID.

    Args:
        model_id: Internal model ID (e.g., "antigravity/claude-sonnet-4")
        context_length: Context window size
        capabilities: List of capabilities (e.g., ["vision", "tools"])

    Returns:
        OllamaModelInfo for /api/tags response
    """
    display_name = generate_model_display_name(model_id)

    return OllamaModelInfo(
        name=display_name,
        model=model_id,
        details=OllamaModelDetails(),
    )


def generate_ollama_show_response(
    model_id: str,
    context_length: int = 128000,
    capabilities: Optional[List[str]] = None,
) -> OllamaShowResponse:
    """
    Generate Ollama /api/show response from internal model ID.

    Args:
        model_id: Internal model ID
        context_length: Context window size
        capabilities: List of capabilities

    Returns:
        OllamaShowResponse
    """
    display_name = generate_model_display_name(model_id)
    caps = capabilities or ["completion"]

    return OllamaShowResponse(
        modelfile=f"FROM {display_name}",
        parameters='stop "<|eot_id|>"',
        template="{{ .Prompt }}",
        details=OllamaModelDetails(),
        model_info={
            "general.architecture": "llama",
            "general.file_type": 2,
            "general.parameter_count": 7000000000,
            "llama.context_length": context_length,
            "llama.embedding_length": 4096,
            "tokenizer.ggml.model": "gpt2",
        },
        capabilities=caps,
    )
