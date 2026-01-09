"""
Streaming wrapper for converting OpenAI streaming format to Ollama streaming format.

This module provides a framework-agnostic streaming wrapper that converts
OpenAI SSE (Server-Sent Events) format to Ollama's NDJSON streaming format.
"""

import json
import logging
from datetime import datetime
from typing import AsyncGenerator, Callable, Optional, Awaitable, Dict, Any

logger = logging.getLogger("rotator_library.ollama_compat")


async def ollama_streaming_wrapper(
    openai_stream: AsyncGenerator[str, None],
    model_name: str,
    is_disconnected: Optional[Callable[[], Awaitable[bool]]] = None,
) -> AsyncGenerator[str, None]:
    """
    Convert OpenAI streaming format to Ollama NDJSON format.

    This is a framework-agnostic wrapper that can be used with any async web framework.
    Instead of SSE format, Ollama uses newline-delimited JSON (NDJSON).

    OpenAI format: data: {...}\n\n
    Ollama format: {...}\n

    Args:
        openai_stream: AsyncGenerator yielding OpenAI SSE format strings
        model_name: The display name to include in responses
        is_disconnected: Optional async callback that returns True if client disconnected

    Yields:
        NDJSON format strings (one JSON object per line)
    """
    accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}
    accumulated_content = ""
    accumulated_thinking = ""
    finish_reason = None
    last_yield_time = datetime.utcnow()
    ping_interval = 10  # seconds

    try:
        async for chunk_str in openai_stream:
            # Check for client disconnection if callback provided
            if is_disconnected is not None and await is_disconnected():
                logger.info("Client disconnected, stopping Ollama stream")
                break

            if not chunk_str.strip():
                continue

            # Skip non-data lines (SSE format)
            if not chunk_str.startswith("data:"):
                continue

            data_content = chunk_str[len("data:"):].strip()

            # Handle [DONE] marker
            if data_content == "[DONE]":
                # Build final tool calls if any
                final_tool_calls = None
                if accumulated_tool_calls:
                    final_tool_calls = []
                    for idx in sorted(accumulated_tool_calls.keys()):
                        tc = accumulated_tool_calls[idx]
                        try:
                            args = json.loads(tc["arguments"])
                        except json.JSONDecodeError:
                            args = {}
                        final_tool_calls.append({
                            "function": {
                                "name": tc["name"],
                                "arguments": args,
                            }
                        })

                # Determine done_reason
                done_reason = "stop"
                if finish_reason == "tool_calls":
                    done_reason = "tool_calls"
                elif finish_reason == "length":
                    done_reason = "length"

                # Send final chunk
                final_chunk = {
                    "model": model_name,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "message": {
                        "role": "assistant",
                        "content": "",
                    },
                    "done": True,
                    "done_reason": done_reason,
                    # Fake performance stats for compatibility
                    "total_duration": 1000000000,
                    "load_duration": 100000000,
                    "prompt_eval_count": 0,
                    "prompt_eval_duration": 100000000,
                    "eval_count": 0,
                    "eval_duration": 800000000,
                }

                if final_tool_calls:
                    final_chunk["message"]["tool_calls"] = final_tool_calls
                if accumulated_thinking:
                    final_chunk["message"]["thinking"] = ""  # Clear, already sent incrementally

                yield json.dumps(final_chunk) + "\n"
                break

            try:
                chunk = json.loads(data_content)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices", [])
            if not choices:
                # Handle usage-only chunks
                continue

            delta = choices[0].get("delta", {})
            chunk_finish_reason = choices[0].get("finish_reason")
            if chunk_finish_reason:
                finish_reason = chunk_finish_reason

            # Extract content
            content = delta.get("content", "")

            # Extract thinking/reasoning content from various provider formats
            thinking = None

            # OpenAI/Anthropic style: reasoning or reasoning_content field
            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
            if isinstance(reasoning, str) and len(reasoning) > 0:
                thinking = reasoning
                accumulated_thinking += reasoning

            # Google Gemini API: extra_content.google.thought is a boolean flag
            # When true, the content field contains the thinking (with <thought> tags)
            extra_content = delta.get("extra_content", {})
            google_thought = extra_content.get("google", {}).get("thought")
            if google_thought is True:
                # Content contains thinking text with <thought> tags
                thinking = content.replace("<thought>", "").replace("</thought>", "")
                accumulated_thinking += thinking
                content = ""  # Clear content since it was actually thinking

            # Handle tool calls
            tool_calls_delta = delta.get("tool_calls", [])
            for tc_chunk in tool_calls_delta:
                index = tc_chunk.get("index", 0)
                if index not in accumulated_tool_calls:
                    accumulated_tool_calls[index] = {
                        "id": tc_chunk.get("id", ""),
                        "name": "",
                        "arguments": "",
                    }

                if tc_chunk.get("id"):
                    accumulated_tool_calls[index]["id"] = tc_chunk["id"]
                if tc_chunk.get("function"):
                    func = tc_chunk["function"]
                    if func.get("name"):
                        accumulated_tool_calls[index]["name"] += func["name"]
                    if func.get("arguments"):
                        accumulated_tool_calls[index]["arguments"] += func["arguments"]

            # Emit chunks - send thinking and content separately (Raycast expectation)
            # When there's thinking, send it with empty content
            if thinking:
                thinking_chunk = {
                    "model": model_name,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "thinking": thinking,
                    },
                    "done": False,
                }
                yield json.dumps(thinking_chunk) + "\n"
                last_yield_time = datetime.utcnow()

            # When there's content (and it's not thinking), send it without thinking field
            if content:
                content_chunk = {
                    "model": model_name,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "done": False,
                }
                yield json.dumps(content_chunk) + "\n"
                last_yield_time = datetime.utcnow()

            # Send periodic pings to keep connection alive
            now = datetime.utcnow()
            if (now - last_yield_time).total_seconds() > ping_interval:
                # Send empty chunk as keepalive
                yield "\n"
                last_yield_time = now

    except Exception as e:
        logger.error(f"Error in Ollama streaming wrapper: {e}")

        # Send error as final chunk
        error_chunk = {
            "model": model_name,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "message": {
                "role": "assistant",
                "content": f"Error: {str(e)}",
            },
            "done": True,
            "done_reason": "stop",
        }
        yield json.dumps(error_chunk) + "\n"
