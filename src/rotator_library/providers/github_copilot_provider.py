# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/providers/github_copilot_provider.py
"""
GitHub Copilot Provider

Provider implementation for GitHub Copilot API, enabling access to
multiple AI models (GPT, Claude, Gemini, etc.) through GitHub Copilot.

Key Features:
- OAuth authentication via GitHub Device Flow
- Support for multiple AI model families
- OpenAI-compatible API translation
- Streaming support

API Endpoints:
- Standard models: POST /chat/completions
- GPT-5/o-series: POST /responses (Responses API)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
import litellm

from ..timeout_config import TimeoutConfig

from .provider_interface import ProviderInterface
from .github_copilot_auth_base import GitHubCopilotAuthBase

lib_logger = logging.getLogger("rotator_library")

# =============================================================================
# API CONFIGURATION
# =============================================================================

# GitHub Copilot API base URLs
COPILOT_API_BASE = "https://api.githubcopilot.com"

# User agent for API requests
USER_AGENT = "LLM-API-Key-Proxy/1.0"

# Required headers for Copilot API calls
COPILOT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Openai-Intent": "conversation-edits",
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Available models via GitHub Copilot
# Based on PRD.md and reference implementation
AVAILABLE_MODELS = [
    # GPT models
    "gpt-5.1-codex",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4o",
    # Claude models
    "claude-haiku-4.5",
    "claude-opus-4",
    # Gemini models
    "gemini-3-flash-preview",
    "gemini-2.0-flash-001",
    # Other models
    "grok-code-fast-1",
    "o3",
    "o4-mini",
]

# Models that require the Responses API instead of Chat Completions
# GPT-5 series and o-series models use the /responses endpoint
RESPONSES_API_MODELS = {
    "gpt-5.1-codex",
    "gpt-5-mini",
    "gpt-5-nano",
    "o3",
    "o4-mini",
}


# =============================================================================
# PROVIDER IMPLEMENTATION
# =============================================================================


class GitHubCopilotProvider(GitHubCopilotAuthBase, ProviderInterface):
    """
    Provider implementation for GitHub Copilot API.

    Inherits OAuth authentication from GitHubCopilotAuthBase and
    implements ProviderInterface for integration with the proxy.
    """

    # =========================================================================
    # PROVIDER CONFIGURATION
    # =========================================================================

    # Provider name for env var lookups (e.g., QUOTA_GROUPS_GITHUB_COPILOT_...)
    provider_env_name: str = "github_copilot"

    # Skip cost calculation - Copilot subscription covers costs
    skip_cost_calculation: bool = True

    # Default rotation mode
    default_rotation_mode: str = "balanced"

    # Tier configuration - GitHub Copilot has a single subscription tier
    tier_priorities = {
        "copilot": 1,  # Standard Copilot subscription
        "copilot-enterprise": 1,  # Enterprise has same priority
    }
    default_tier_priority: int = 1

    def __init__(self):
        GitHubCopilotAuthBase.__init__(self)
        # Provider-specific initialization
        self._api_base = COPILOT_API_BASE

    # =========================================================================
    # PROVIDER INTERFACE IMPLEMENTATION
    # =========================================================================

    def has_custom_logic(self) -> bool:
        """
        Returns True as GitHub Copilot uses custom API translation.

        GitHub Copilot has its own API format that requires custom handling
        rather than using litellm's standard provider support.
        """
        return True

    async def get_models(
        self, api_key: str, client: httpx.AsyncClient  # noqa: ARG002
    ) -> List[str]:
        """
        Fetch available models from GitHub Copilot.

        For now, returns a hardcoded list of known available models.
        Future implementations may query the API for dynamic model discovery.

        Args:
            api_key: The credential path (not used for hardcoded list)
            client: HTTP client instance (not used for hardcoded list)

        Returns:
            List of model names prefixed with 'github_copilot/'
        """
        # Unused parameters (required by interface): api_key, client
        del api_key, client  # Silence unused parameter warnings
        # Return hardcoded model list with provider prefix
        return [f"github_copilot/{model}" for model in AVAILABLE_MODELS]

    def get_credential_tier_name(self, credential: str) -> str:
        """
        Returns the tier name for a credential.

        GitHub Copilot credentials are either standard or enterprise.

        Args:
            credential: The credential path

        Returns:
            Tier name string
        """
        # Check if credential has enterprise metadata
        cached = self._credentials_cache.get(credential)
        if cached:
            metadata = cached.get("_proxy_metadata", {})
            if metadata.get("enterprise_domain"):
                return "copilot-enterprise"
        return "copilot"

    def _get_api_base(self, credential_path: str) -> str:
        """
        Get the API base URL for a credential.

        Enterprise credentials use a different base URL.

        Args:
            credential_path: Path to the credential file

        Returns:
            API base URL string
        """
        cached = self._credentials_cache.get(credential_path)
        if cached:
            metadata = cached.get("_proxy_metadata", {})
            enterprise_domain = metadata.get("enterprise_domain")
            if enterprise_domain:
                # Enterprise uses copilot-api.{domain}
                return f"https://copilot-api.{enterprise_domain}"
        return COPILOT_API_BASE

    def _is_responses_api_model(self, model: str) -> bool:
        """
        Check if a model requires the Responses API.

        GPT-5 series and o-series models use /responses instead of /chat/completions.

        Args:
            model: Model name (with or without provider prefix)

        Returns:
            True if the model uses Responses API
        """
        # Strip provider prefix if present
        clean_model = model.split("/")[-1] if "/" in model else model
        return clean_model in RESPONSES_API_MODELS

    # =========================================================================
    # CHAT COMPLETIONS HELPERS
    # =========================================================================

    def _detect_vision_content(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Detect if messages contain vision/image content.

        Checks both Chat Completions format (image_url) and Responses API
        format (input_image) for multimodal content.

        Args:
            messages: List of message dictionaries

        Returns:
            True if any message contains image content
        """
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        # Chat Completions format
                        if part.get("type") == "image_url":
                            return True
                        # Responses API format
                        if part.get("type") == "input_image":
                            return True
        return False

    def _detect_agent_initiated(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Detect if the conversation is agent-initiated.

        A conversation is agent-initiated if the last message is not from a user.

        Args:
            messages: List of message dictionaries

        Returns:
            True if the last message is not from a user
        """
        if not messages:
            return False
        last_msg = messages[-1]
        return last_msg.get("role") != "user"

    async def _build_copilot_headers(
        self,
        credential_path: str,
        is_vision: bool = False,
        is_agent: bool = False,
    ) -> Dict[str, str]:
        """
        Build headers for GitHub Copilot API requests.

        Args:
            credential_path: Path to the credential file
            is_vision: True if request contains vision content
            is_agent: True if request is agent-initiated

        Returns:
            Dictionary of headers for the API request
        """
        auth_header = await self.get_auth_header(credential_path)

        headers = {
            **auth_header,
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            "Openai-Intent": "conversation-edits",
            "x-initiator": "agent" if is_agent else "user",
        }

        if is_vision:
            headers["Copilot-Vision-Request"] = "true"

        return headers

    # =========================================================================
    # ACOMPLETION IMPLEMENTATION
    # =========================================================================

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Handle chat completion request for GitHub Copilot.

        Routes to the appropriate endpoint based on model:
        - Standard models: /chat/completions
        - GPT-5/o-series: /responses (Responses API) - NOT YET IMPLEMENTED

        Args:
            client: HTTP client instance
            **kwargs: Completion parameters including:
                - model: Model name
                - messages: List of messages
                - stream: Whether to stream the response
                - credential_identifier: Path to credential file

        Returns:
            ModelResponse or AsyncGenerator for streaming
        """
        # Extract parameters
        model = kwargs.get("model", "gpt-4o")
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)
        credential_path = kwargs.pop(
            "credential_identifier", kwargs.get("credential_path", "")
        )

        # Normalize model name (strip provider prefix)
        if "/" in model:
            model = model.split("/", 1)[1]

        # Detect content types
        is_vision = self._detect_vision_content(messages)
        is_agent = self._detect_agent_initiated(messages)

        # Get API base URL
        api_base = self._get_api_base(credential_path)

        # Build headers
        headers = await self._build_copilot_headers(
            credential_path, is_vision=is_vision, is_agent=is_agent
        )

        # Check if this model uses Responses API
        if self._is_responses_api_model(model):
            lib_logger.debug(f"Model {model} using Responses API endpoint")
            return await self._responses_api_completion(
                client=client,
                api_base=api_base,
                headers=headers,
                model=model,
                messages=messages,
                stream=stream,
                **kwargs,
            )

        # Build request payload for Chat Completions API
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        # Copy over optional parameters
        optional_params = [
            "temperature",
            "top_p",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
            "stop",
        ]
        for param in optional_params:
            if param in kwargs and kwargs[param] is not None:
                payload[param] = kwargs[param]

        endpoint = f"{api_base}/chat/completions"

        lib_logger.debug(
            f"Copilot request to {model}: {json.dumps(payload, default=str)[:500]}..."
        )

        if stream:
            return self._stream_chat_response(
                client, endpoint, headers, payload, model
            )
        else:
            return await self._non_stream_chat_response(
                client, endpoint, headers, payload, model
            )

    async def _non_stream_chat_response(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
    ) -> litellm.ModelResponse:
        """
        Handle non-streaming chat completion response.

        Args:
            client: HTTP client instance
            endpoint: API endpoint URL
            headers: Request headers
            payload: Request payload
            model: Model name for response

        Returns:
            litellm.ModelResponse with the completion
        """
        response = await client.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=TimeoutConfig.non_streaming(),
        )

        if response.status_code >= 400:
            error_text = response.text
            lib_logger.error(
                f"Copilot API error {response.status_code}: {error_text[:500]}"
            )
            raise httpx.HTTPStatusError(
                f"Copilot API error: {response.status_code}",
                request=response.request,
                response=response,
            )

        data = response.json()

        # Translate to litellm format
        created = data.get("created", int(time.time()))
        response_id = data.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}")

        choices = []
        for choice in data.get("choices", []):
            message = choice.get("message", {})
            choices.append(
                {
                    "index": choice.get("index", 0),
                    "message": {
                        "role": message.get("role", "assistant"),
                        "content": message.get("content"),
                    },
                    "finish_reason": choice.get("finish_reason", "stop"),
                }
            )

        # Handle tool calls if present
        for i, choice in enumerate(data.get("choices", [])):
            message = choice.get("message", {})
            if "tool_calls" in message:
                choices[i]["message"]["tool_calls"] = message["tool_calls"]

        response_obj = litellm.ModelResponse(
            id=response_id,
            created=created,
            model=f"github_copilot/{model}",
            object="chat.completion",
            choices=choices,
        )

        # Add usage if available
        if "usage" in data:
            usage = data["usage"]
            response_obj.usage = litellm.Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )

        return response_obj

    async def _stream_chat_response(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """
        Handle streaming chat completion response.

        Parses SSE chunks from the API and yields litellm ModelResponse chunks.

        Args:
            client: HTTP client instance
            endpoint: API endpoint URL
            headers: Request headers
            payload: Request payload
            model: Model name for response

        Yields:
            litellm.ModelResponse chunks
        """
        created = int(time.time())
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        # Track tool calls state
        current_tool_calls: Dict[int, Dict[str, Any]] = {}

        async with client.stream(
            "POST",
            endpoint,
            headers=headers,
            json=payload,
            timeout=TimeoutConfig.streaming(),
        ) as response:
            if response.status_code >= 400:
                error_body = await response.aread()
                error_text = error_body.decode("utf-8", errors="ignore")
                lib_logger.error(
                    f"Copilot API error {response.status_code}: {error_text[:500]}"
                )
                raise httpx.HTTPStatusError(
                    f"Copilot API error: {response.status_code}",
                    request=response.request,
                    response=response,
                )

            async for line in response.aiter_lines():
                if not line:
                    continue

                if not line.startswith("data: "):
                    continue

                data = line[6:].strip()
                if not data or data == "[DONE]":
                    continue

                try:
                    evt = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # Extract response ID from first chunk
                if evt.get("id"):
                    response_id = evt["id"]

                # Process choices
                for choice in evt.get("choices", []):
                    index = choice.get("index", 0)
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason")

                    # Handle content delta
                    if "content" in delta:
                        chunk = litellm.ModelResponse(
                            id=response_id,
                            created=created,
                            model=f"github_copilot/{model}",
                            object="chat.completion.chunk",
                            choices=[
                                {
                                    "index": index,
                                    "delta": {
                                        "content": delta["content"],
                                        "role": delta.get("role", "assistant"),
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        )
                        yield chunk

                    # Handle tool calls delta
                    if "tool_calls" in delta:
                        for tc in delta["tool_calls"]:
                            tc_index = tc.get("index", 0)

                            if tc_index not in current_tool_calls:
                                current_tool_calls[tc_index] = {
                                    "id": tc.get("id", ""),
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }

                            # Accumulate function data
                            if "id" in tc and tc["id"]:
                                current_tool_calls[tc_index]["id"] = tc["id"]
                            if "function" in tc:
                                func = tc["function"]
                                if "name" in func:
                                    current_tool_calls[tc_index]["function"][
                                        "name"
                                    ] += func["name"]
                                if "arguments" in func:
                                    current_tool_calls[tc_index]["function"][
                                        "arguments"
                                    ] += func["arguments"]

                    # Handle finish
                    if finish_reason:
                        # Send accumulated tool calls if any
                        if current_tool_calls and finish_reason == "tool_calls":
                            tool_calls_list = [
                                current_tool_calls[i]
                                for i in sorted(current_tool_calls.keys())
                            ]
                            chunk = litellm.ModelResponse(
                                id=response_id,
                                created=created,
                                model=f"github_copilot/{model}",
                                object="chat.completion.chunk",
                                choices=[
                                    {
                                        "index": index,
                                        "delta": {"tool_calls": tool_calls_list},
                                        "finish_reason": None,
                                    }
                                ],
                            )
                            yield chunk

                        # Final chunk
                        final_chunk = litellm.ModelResponse(
                            id=response_id,
                            created=created,
                            model=f"github_copilot/{model}",
                            object="chat.completion.chunk",
                            choices=[
                                {
                                    "index": index,
                                    "delta": {},
                                    "finish_reason": finish_reason,
                                }
                            ],
                        )
                        yield final_chunk

    # =========================================================================
    # RESPONSES API IMPLEMENTATION
    # =========================================================================

    def _convert_messages_to_responses_format(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert Chat Completions messages to Responses API input format.

        The Responses API uses a different message structure:
        - 'system' role becomes 'developer'
        - Content can be structured with type: 'input_text' or 'input_image'
        - Previous assistant responses use type: 'message' with 'output_text'

        Args:
            messages: List of Chat Completions format messages

        Returns:
            List of Responses API format input items
        """
        input_items = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            # Map system role to developer (Responses API convention)
            if role == "system":
                role = "developer"

            # Handle assistant messages (previous responses)
            if role == "assistant":
                # Convert to Responses API message format
                if isinstance(content, str):
                    input_items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content}],
                    })
                elif isinstance(content, list):
                    # Convert content parts
                    output_content = []
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                output_content.append({
                                    "type": "output_text",
                                    "text": part.get("text", ""),
                                })
                            else:
                                output_content.append(part)
                        else:
                            output_content.append({
                                "type": "output_text",
                                "text": str(part),
                            })
                    input_items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": output_content,
                    })
                continue

            # Handle user/developer messages
            if isinstance(content, str):
                # Simple string content
                input_items.append({
                    "role": role,
                    "content": [{"type": "input_text", "text": content}],
                })
            elif isinstance(content, list):
                # Structured content - convert types
                converted_content = []
                for part in content:
                    if isinstance(part, dict):
                        part_type = part.get("type", "")
                        if part_type == "text":
                            converted_content.append({
                                "type": "input_text",
                                "text": part.get("text", ""),
                            })
                        elif part_type == "image_url":
                            # Convert image_url to input_image format
                            image_url = part.get("image_url", {})
                            url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                            converted_content.append({
                                "type": "input_image",
                                "image_url": url,
                            })
                        else:
                            # Pass through other types
                            converted_content.append(part)
                    else:
                        converted_content.append({
                            "type": "input_text",
                            "text": str(part),
                        })
                input_items.append({
                    "role": role,
                    "content": converted_content,
                })
            else:
                # Fallback for unexpected content types
                input_items.append({
                    "role": role,
                    "content": [{"type": "input_text", "text": str(content) if content else ""}],
                })

        return input_items

    async def _responses_api_completion(
        self,
        client: httpx.AsyncClient,
        api_base: str,
        headers: Dict[str, str],
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool,
        **kwargs,
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Handle completion request using the Responses API.

        The Responses API is used for GPT-5 and o-series models.

        Args:
            client: HTTP client instance
            api_base: API base URL
            headers: Request headers
            model: Model name
            messages: List of messages in Chat Completions format
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            ModelResponse or AsyncGenerator for streaming
        """
        # Convert messages to Responses API format
        input_items = self._convert_messages_to_responses_format(messages)

        # Build request payload
        payload: Dict[str, Any] = {
            "model": model,
            "input": input_items,
            "stream": stream,
        }

        # Copy over optional parameters with Responses API naming
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            payload["max_output_tokens"] = kwargs["max_tokens"]
        if "temperature" in kwargs and kwargs["temperature"] is not None:
            payload["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs and kwargs["top_p"] is not None:
            payload["top_p"] = kwargs["top_p"]

        endpoint = f"{api_base}/responses"

        lib_logger.debug(
            f"Copilot Responses API request to {model}: {json.dumps(payload, default=str)[:500]}..."
        )

        if stream:
            return self._stream_responses_api(client, endpoint, headers, payload, model)
        else:
            return await self._non_stream_responses_api(
                client, endpoint, headers, payload, model
            )

    async def _non_stream_responses_api(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
    ) -> litellm.ModelResponse:
        """
        Handle non-streaming Responses API response.

        Args:
            client: HTTP client instance
            endpoint: API endpoint URL
            headers: Request headers
            payload: Request payload
            model: Model name for response

        Returns:
            litellm.ModelResponse with the completion
        """
        response = await client.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=TimeoutConfig.non_streaming(),
        )

        if response.status_code >= 400:
            error_text = response.text
            lib_logger.error(
                f"Copilot Responses API error {response.status_code}: {error_text[:500]}"
            )
            raise httpx.HTTPStatusError(
                f"Copilot Responses API error: {response.status_code}",
                request=response.request,
                response=response,
            )

        data = response.json()

        # Translate Responses API format to litellm format
        created = data.get("created_at", int(time.time()))
        if isinstance(created, str):
            # Parse ISO timestamp if needed
            try:
                from datetime import datetime
                created = int(datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp())
            except (ValueError, TypeError):
                created = int(time.time())

        response_id = data.get("id", f"resp-{uuid.uuid4().hex[:8]}")

        # Extract output text from the response
        output_text = data.get("output_text", "")

        # If output_text is not directly available, extract from output array
        if not output_text and "output" in data:
            for item in data.get("output", []):
                if item.get("type") == "message" and item.get("role") == "assistant":
                    for content_part in item.get("content", []):
                        if content_part.get("type") == "output_text":
                            output_text = content_part.get("text", "")
                            break
                    if output_text:
                        break

        # Build choices in Chat Completions format
        choices = [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output_text,
                },
                "finish_reason": data.get("status", "completed") == "completed" and "stop" or "stop",
            }
        ]

        response_obj = litellm.ModelResponse(
            id=response_id,
            created=created,
            model=f"github_copilot/{model}",
            object="chat.completion",
            choices=choices,
        )

        # Add usage if available
        if "usage" in data:
            usage = data["usage"]
            response_obj.usage = litellm.Usage(
                prompt_tokens=usage.get("input_tokens", usage.get("prompt_tokens", 0)),
                completion_tokens=usage.get("output_tokens", usage.get("completion_tokens", 0)),
                total_tokens=usage.get("total_tokens", 0),
            )

        return response_obj

    async def _stream_responses_api(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """
        Handle streaming Responses API response.

        Parses SSE chunks from the API and yields litellm ModelResponse chunks.

        Args:
            client: HTTP client instance
            endpoint: API endpoint URL
            headers: Request headers
            payload: Request payload
            model: Model name for response

        Yields:
            litellm.ModelResponse chunks
        """
        created = int(time.time())
        response_id = f"resp-{uuid.uuid4().hex[:8]}"

        async with client.stream(
            "POST",
            endpoint,
            headers=headers,
            json=payload,
            timeout=TimeoutConfig.streaming(),
        ) as response:
            if response.status_code >= 400:
                error_body = await response.aread()
                error_text = error_body.decode("utf-8", errors="ignore")
                lib_logger.error(
                    f"Copilot Responses API error {response.status_code}: {error_text[:500]}"
                )
                raise httpx.HTTPStatusError(
                    f"Copilot Responses API error: {response.status_code}",
                    request=response.request,
                    response=response,
                )

            async for line in response.aiter_lines():
                if not line:
                    continue

                if not line.startswith("data: "):
                    continue

                data = line[6:].strip()
                if not data or data == "[DONE]":
                    continue

                try:
                    evt = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # Extract response ID from first event
                if evt.get("response", {}).get("id"):
                    response_id = evt["response"]["id"]
                elif evt.get("id"):
                    response_id = evt["id"]

                # Handle different event types
                event_type = evt.get("type", "")

                # Handle content delta events
                if event_type == "response.output_text.delta":
                    delta_text = evt.get("delta", "")
                    if delta_text:
                        chunk = litellm.ModelResponse(
                            id=response_id,
                            created=created,
                            model=f"github_copilot/{model}",
                            object="chat.completion.chunk",
                            choices=[
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": delta_text,
                                        "role": "assistant",
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        )
                        yield chunk

                # Handle response done event
                elif event_type == "response.done":
                    final_chunk = litellm.ModelResponse(
                        id=response_id,
                        created=created,
                        model=f"github_copilot/{model}",
                        object="chat.completion.chunk",
                        choices=[
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }
                        ],
                    )
                    yield final_chunk

                # Handle content_part delta (alternative format)
                elif event_type == "response.content_part.delta":
                    content_part = evt.get("part", {})
                    if content_part.get("type") == "output_text":
                        delta_text = content_part.get("text", "")
                        if delta_text:
                            chunk = litellm.ModelResponse(
                                id=response_id,
                                created=created,
                                model=f"github_copilot/{model}",
                                object="chat.completion.chunk",
                                choices=[
                                    {
                                        "index": 0,
                                        "delta": {
                                            "content": delta_text,
                                            "role": "assistant",
                                        },
                                        "finish_reason": None,
                                    }
                                ],
                            )
                            yield chunk
