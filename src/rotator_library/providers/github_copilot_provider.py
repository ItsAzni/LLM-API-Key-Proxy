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

# OpenCode version cache (fetched once on first use)
_opencode_version_cache: Optional[str] = None
_OPENCODE_FALLBACK_VERSION = "1.1.34"  # Fallback if GitHub fetch fails


def _fetch_opencode_version() -> str:
    """
    Fetch the latest OpenCode version from GitHub releases.

    Caches the result so it only fetches once per process lifetime.
    Falls back to a hardcoded version if the fetch fails.

    Returns:
        Version string like "1.1.34"
    """
    global _opencode_version_cache

    if _opencode_version_cache is not None:
        return _opencode_version_cache

    try:
        import urllib.request
        import json as json_module

        url = "https://api.github.com/repos/anomalyco/opencode/releases/latest"
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "LLM-API-Key-Proxy",
            },
        )

        with urllib.request.urlopen(req, timeout=5) as response:
            data = json_module.loads(response.read().decode())
            tag = data.get("tag_name", "")
            # Strip leading 'v' if present (e.g., "v1.0.220" -> "1.0.220")
            version = tag.lstrip("v") if tag else _OPENCODE_FALLBACK_VERSION
            _opencode_version_cache = version
            lib_logger.debug(f"Fetched OpenCode version from GitHub: {version}")
            return version

    except Exception as e:
        lib_logger.warning(
            f"Failed to fetch OpenCode version from GitHub: {e}. "
            f"Using fallback version {_OPENCODE_FALLBACK_VERSION}"
        )
        _opencode_version_cache = _OPENCODE_FALLBACK_VERSION
        return _OPENCODE_FALLBACK_VERSION


def _get_user_agent() -> str:
    """Get the User-Agent string mimicking OpenCode."""
    version = _fetch_opencode_version()
    return f"opencode/{version}"


# User agent for API requests (dynamically fetched from GitHub)
USER_AGENT = _get_user_agent()

# Required headers for Copilot API calls
COPILOT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Openai-Intent": "conversation-edits",
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Available models via GitHub Copilot
# Based on models.dev database and opencode reference
AVAILABLE_MODELS = [
    # GPT-5 series (reasoning models - use Responses API)
    "gpt-5",
    "gpt-5.1",
    "gpt-5.1-codex",
    "gpt-5.1-codex-mini",
    "gpt-5.1-codex-max",
    "gpt-5.2",
    "gpt-5-codex",
    # GPT-5 non-reasoning (use Chat Completions API)
    "gpt-5-mini",
    "gpt-5-nano",
    # GPT-4 series
    "gpt-4o",
    "gpt-4.1",
    # o-series (reasoning models - use Responses API)
    "o3",
    "o3-mini",
    "o4-mini",
    # Claude models
    "claude-sonnet-4",
    "claude-sonnet-4.5",
    "claude-opus-4",
    "claude-opus-4.5",
    "claude-opus-41",
    "claude-haiku-4.5",
    "claude-3.5-sonnet",
    "claude-3.7-sonnet",
    "claude-3.7-sonnet-thought",
    # Gemini models
    "gemini-2.0-flash-001",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    # Other models
    "grok-code-fast-1",
]


def _is_responses_api_model(model: str) -> bool:
    """
    Check if a model requires the Responses API.

    Based on opencode's shouldUseCopilotResponsesApi logic (provider.ts:52-54):
    - GPT-5 and later use Responses API, EXCEPT gpt-5-mini
    - Note: gpt-5-nano DOES use Responses API (unlike gpt-5-mini)

    Args:
        model: Model name (with or without provider prefix)

    Returns:
        True if the model uses Responses API
    """
    # Strip provider prefix if present
    clean_model = model.split("/")[-1] if "/" in model else model

    # Check for GPT-5 or later (matching OpenCode's isGpt5OrLater + shouldUseCopilotResponsesApi)
    import re

    match = re.match(r"^gpt-(\d+)", clean_model)
    if match:
        version = int(match.group(1))
        if version >= 5:
            # Only gpt-5-mini uses Chat Completions, not Responses API
            # Note: gpt-5-nano DOES use Responses API per OpenCode
            if clean_model.startswith("gpt-5-mini"):
                return False
            return True

    return False


def _is_claude_model(model: str) -> bool:
    """Check if model is a Claude model."""
    clean = model.split("/")[-1].lower() if "/" in model else model.lower()
    return clean.startswith("claude")


def _is_gemini_model(model: str) -> bool:
    """Check if model is a Gemini model."""
    clean = model.split("/")[-1].lower() if "/" in model else model.lower()
    return clean.startswith("gemini")


def _is_gpt5_or_o_series(model: str) -> bool:
    """Check if model is GPT-5 or o-series (reasoning models)."""
    clean = model.split("/")[-1].lower() if "/" in model else model.lower()
    if clean.startswith("o3") or clean.startswith("o4"):
        return True
    import re

    match = re.match(r"^gpt-(\d+)", clean)
    if match and int(match.group(1)) >= 5:
        return True
    return False


def _map_reasoning_effort_to_config(
    reasoning_effort: Optional[str],
    model: str,
) -> Dict[str, Any]:
    """
    Map reasoning_effort to model-specific thinking configuration.

    Similar to Antigravity provider's _get_thinking_config.

    Args:
        reasoning_effort: Effort level (low, medium, high, etc.)
        model: Model name

    Returns:
        Dict with model-specific thinking parameters
    """
    clean_model = model.split("/")[-1] if "/" in model else model

    # Normalize effort
    if reasoning_effort is None:
        effort = "auto"
    elif isinstance(reasoning_effort, str):
        effort = reasoning_effort.strip().lower() or "auto"
    else:
        effort = "auto"

    valid_efforts = {
        "auto",
        "disable",
        "off",
        "none",
        "minimal",
        "low",
        "low_medium",
        "medium",
        "medium_high",
        "high",
        "xhigh",
        "max",
    }
    if effort not in valid_efforts:
        lib_logger.warning(
            f"[GitHubCopilot] Unknown reasoning_effort: '{reasoning_effort}', using auto"
        )
        effort = "auto"

    # GPT-5 and o-series: reasoning.effort (none/minimal/low/medium/high/xhigh)
    if _is_gpt5_or_o_series(clean_model):
        if effort in ("disable", "off", "none"):
            return {}  # No reasoning
        if effort == "minimal":
            return {"reasoning": {"effort": "minimal", "summary": "auto"}}
        if effort == "low":
            return {"reasoning": {"effort": "low", "summary": "auto"}}
        if effort in ("low_medium", "medium"):
            return {"reasoning": {"effort": "medium", "summary": "auto"}}
        if effort in ("medium_high", "high"):
            return {"reasoning": {"effort": "high", "summary": "auto"}}
        if effort in ("xhigh", "max"):
            return {"reasoning": {"effort": "xhigh", "summary": "auto"}}
        # auto - default to high
        return {"reasoning": {"effort": "high", "summary": "auto"}}

    # Claude models: thinking_budget (tokens)
    if _is_claude_model(clean_model):
        if effort in ("disable", "off", "none"):
            return {}  # No thinking
        budgets = {
            "auto": 16000,  # Middle ground
            "minimal": 1024,
            "low": 4096,
            "low_medium": 8192,
            "medium": 16000,
            "medium_high": 24000,
            "high": 32000,
        }
        return {"thinking": {"budget_tokens": budgets.get(effort, 16000)}}

    # Gemini models: thinkingLevel or thinkingBudget
    if _is_gemini_model(clean_model):
        is_gemini_3 = "gemini-3" in clean_model

        if is_gemini_3:
            # Gemini 3: thinkingLevel (minimal/low/medium/high)
            if effort in ("disable", "off", "none"):
                return {"thinkingConfig": {"thinkingLevel": "minimal"}}
            if effort in ("minimal", "low"):
                return {"thinkingConfig": {"thinkingLevel": "low"}}
            if effort in ("low_medium", "medium"):
                return {"thinkingConfig": {"thinkingLevel": "medium"}}
            return {"thinkingConfig": {"thinkingLevel": "high"}}
        else:
            # Gemini 2.5: thinkingBudget (tokens)
            if effort in ("disable", "off", "none"):
                return {"thinkingConfig": {"thinkingBudget": 0}}
            budgets = {
                "auto": -1,  # Auto
                "minimal": 3072,
                "low": 6144,
                "low_medium": 9216,
                "medium": 12288,
                "medium_high": 18432,
                "high": 24576,
            }
            return {"thinkingConfig": {"thinkingBudget": budgets.get(effort, -1)}}

    return {}


def _convert_tools_to_responses_format(
    tools: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Convert OpenAI-format tools to Responses API format.

    Handles special tool types like web_search and web_search_preview,
    which are passed directly without function wrapping.

    Args:
        tools: List of OpenAI-format tools

    Returns:
        List of Responses API format tools
    """
    if not tools:
        return []

    responses_tools = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue

        tool_type = tool.get("type", "function")

        if tool_type == "function":
            func = tool.get("function", {})
            name = func.get("name", "")
            # Skip tools without a name
            if not name:
                continue
            params = func.get("parameters", {})
            # Ensure parameters is a valid object
            if not isinstance(params, dict):
                params = {"type": "object", "properties": {}}
            responses_tools.append(
                {
                    "type": "function",
                    "name": name,
                    "description": func.get("description") or "",
                    "parameters": params,
                    "strict": False,
                }
            )
        elif tool_type in ("web_search", "web_search_preview"):
            # Web search is a special built-in tool type
            responses_tools.append({"type": tool_type})
        elif tool_type == "code_interpreter":
            # Code interpreter is another built-in tool type
            responses_tools.append({"type": "code_interpreter"})
        elif tool_type == "file_search":
            # File search built-in tool
            responses_tools.append({"type": "file_search"})
        else:
            # Unknown tool types - pass through as-is
            responses_tools.append(tool)

    return responses_tools


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

    # Cache for discovered models: {model_id: supported_endpoints}
    _discovered_models: Dict[str, List[str]] = {}

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetch available models from GitHub Copilot's /models endpoint.

        Dynamically discovers models and their supported endpoints.
        Falls back to hardcoded list if API call fails.

        Args:
            api_key: The credential path for authentication
            client: HTTP client instance

        Returns:
            List of model names prefixed with 'github_copilot/'
        """
        try:
            # Get auth header
            auth_header = await self.get_auth_header(api_key)
            headers = {
                **auth_header,
                "User-Agent": USER_AGENT,
            }

            # Fetch models from API
            api_base = self._get_api_base(api_key)
            response = await client.get(
                f"{api_base}/models",
                headers=headers,
                timeout=10.0,
            )

            if response.status_code == 200:
                data = response.json()
                models = []

                for model in data.get("data", []):
                    model_id = model.get("id")
                    if not model_id:
                        continue

                    # Skip embedding models
                    if "embedding" in model_id.lower():
                        continue

                    # Cache supported endpoints for this model
                    supported_endpoints = model.get("supported_endpoints", [])
                    self._discovered_models[model_id] = supported_endpoints

                    models.append(f"github_copilot/{model_id}")

                if models:
                    lib_logger.debug(
                        f"Discovered {len(models)} models from GitHub Copilot API"
                    )
                    return models

        except Exception as e:
            lib_logger.warning(
                f"Failed to fetch models from GitHub Copilot API: {e}. Using fallback list."
            )

        # Fallback to hardcoded list
        return [f"github_copilot/{model}" for model in AVAILABLE_MODELS]

    def _get_model_endpoints(self, model: str) -> List[str]:
        """
        Get the supported endpoints for a model from discovered data.

        Args:
            model: Model name (with or without provider prefix)

        Returns:
            List of supported endpoints (e.g., ['/responses', '/chat/completions'])
        """
        clean_model = model.split("/")[-1] if "/" in model else model
        return self._discovered_models.get(clean_model, [])

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

        Based on OpenCode's shouldUseCopilotResponsesApi logic:
        - GPT-5 and later use Responses API (for reasoning support)
        - EXCEPT gpt-5-mini and gpt-5-nano which use Chat Completions
        - o3 and o4-mini also use Responses API

        Args:
            model: Model name (with or without provider prefix)

        Returns:
            True if the model uses Responses API
        """
        # Always use the heuristic based on model name patterns
        # This matches OpenCode behavior: GPT-5+ uses Responses API for reasoning
        return _is_responses_api_model(model)

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

    # Responses API alternate input types that indicate agent-initiated requests
    # Based on opencode copilot-auth plugin
    RESPONSES_API_AGENT_TYPES = {
        "file_search_call",
        "computer_call",
        "computer_call_output",
        "web_search_call",
        "function_call",
        "function_call_output",
        "image_generation_call",
        "code_interpreter_call",
        "local_shell_call",
        "local_shell_call_output",
        "mcp_list_tools",
        "mcp_approval_request",
        "mcp_approval_response",
        "mcp_call",
        "reasoning",
    }

    def _detect_agent_initiated(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Detect if the conversation is agent-initiated.

        Based on opencode's copilot-force-agent-header plugin:
        - Returns True if ANY message has role "assistant" or "tool"
        - Also checks for Responses API alternate input types

        This enables better agentic behavior from GitHub Copilot models.

        Args:
            messages: List of message dictionaries

        Returns:
            True if conversation contains assistant/tool messages
        """
        import os

        # Force agent mode via environment variable (default: false to match opencode behavior)
        # When false, x-initiator is set dynamically based on whether last message is from user
        force_agent = os.getenv("GITHUB_COPILOT_FORCE_AGENT", "false").lower() not in (
            "false",
            "0",
            "no",
        )
        if force_agent:
            return True

        if not messages:
            return False

        # Match OpenCode's behavior: only check if the LAST message's role is NOT "user"
        # This means:
        # - First user message → x-initiator: "user"
        # - User message after assistant response → x-initiator: "user" (new user turn)
        # - Tool result after function call → x-initiator: "agent" (continuing agent flow)
        last_msg = messages[-1]
        last_role = last_msg.get("role", "user")

        # For Chat Completions format: check last message role
        if last_role != "user":
            return True

        # Check Responses API format: last input with agent type
        msg_type = last_msg.get("type", "")
        if msg_type in self.RESPONSES_API_AGENT_TYPES:
            return True

        return False

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
            # Remove keys we pass explicitly to avoid duplicate keyword args
            filtered_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                not in (
                    "model",
                    "messages",
                    "stream",
                    "credential_identifier",
                    "credential_path",
                )
            }
            return await self._responses_api_completion(
                client=client,
                api_base=api_base,
                headers=headers,
                model=model,
                messages=messages,
                stream=stream,
                **filtered_kwargs,
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
            # Tool/function calling
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            # Structured output
            "response_format",
        ]
        for param in optional_params:
            if param in kwargs and kwargs[param] is not None:
                payload[param] = kwargs[param]

        # Map reasoning_effort to model-specific thinking config
        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort is not None:
            thinking_config = _map_reasoning_effort_to_config(reasoning_effort, model)
            payload.update(thinking_config)
            lib_logger.debug(
                "[GitHubCopilot] reasoning_effort=%s mapped_config=%s",
                reasoning_effort,
                json.dumps(thinking_config, default=str)[:500],
            )

        endpoint = f"{api_base}/chat/completions"

        lib_logger.debug(
            f"Copilot request to {model}: {json.dumps(payload, default=str)[:500]}..."
        )

        if stream:
            return self._stream_chat_response(client, endpoint, headers, payload, model)
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

                    # Handle reasoning_content delta (for models that support reasoning via Chat Completions)
                    if "reasoning_content" in delta:
                        chunk = litellm.ModelResponse(
                            id=response_id,
                            created=created,
                            model=f"github_copilot/{model}",
                            object="chat.completion.chunk",
                            choices=[
                                {
                                    "index": index,
                                    "delta": {
                                        "reasoning_content": delta["reasoning_content"],
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
                    input_items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": content}],
                        }
                    )
                elif isinstance(content, list):
                    # Convert content parts
                    output_content = []
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                output_content.append(
                                    {
                                        "type": "output_text",
                                        "text": part.get("text", ""),
                                    }
                                )
                            else:
                                output_content.append(part)
                        else:
                            output_content.append(
                                {
                                    "type": "output_text",
                                    "text": str(part),
                                }
                            )
                    input_items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": output_content,
                        }
                    )
                tool_calls = msg.get("tool_calls") or []
                for tool_call in tool_calls:
                    if tool_call.get("type") != "function":
                        continue
                    function = tool_call.get("function", {})
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": tool_call.get("id", ""),
                            "name": function.get("name", ""),
                            "arguments": function.get("arguments", ""),
                        }
                    )
                continue

            # Handle tool messages (function outputs)
            if role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                if isinstance(content, str):
                    output = content
                elif content is None:
                    output = ""
                else:
                    output = json.dumps(content)
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": output,
                    }
                )
                continue

            # Handle user/developer messages
            if isinstance(content, str):
                # Simple string content
                input_items.append(
                    {
                        "role": role,
                        "content": [{"type": "input_text", "text": content}],
                    }
                )
            elif isinstance(content, list):
                # Structured content - convert types
                converted_content = []
                for part in content:
                    if isinstance(part, dict):
                        part_type = part.get("type", "")
                        if part_type == "text":
                            converted_content.append(
                                {
                                    "type": "input_text",
                                    "text": part.get("text", ""),
                                }
                            )
                        elif part_type == "image_url":
                            # Convert image_url to input_image format
                            image_url = part.get("image_url", {})
                            url = (
                                image_url.get("url", "")
                                if isinstance(image_url, dict)
                                else str(image_url)
                            )
                            converted_content.append(
                                {
                                    "type": "input_image",
                                    "image_url": url,
                                }
                            )
                        else:
                            # Pass through other types
                            converted_content.append(part)
                    else:
                        converted_content.append(
                            {
                                "type": "input_text",
                                "text": str(part),
                            }
                        )
                input_items.append(
                    {
                        "role": role,
                        "content": converted_content,
                    }
                )
            else:
                # Fallback for unexpected content types
                input_items.append(
                    {
                        "role": role,
                        "content": [
                            {
                                "type": "input_text",
                                "text": str(content) if content else "",
                            }
                        ],
                    }
                )

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

        # Tool/function calling support - convert to Responses API format
        if "tools" in kwargs and kwargs["tools"] is not None:
            converted_tools = _convert_tools_to_responses_format(kwargs["tools"])
            if converted_tools:
                payload["tools"] = converted_tools
        if "tool_choice" in kwargs and kwargs["tool_choice"] is not None:
            payload["tool_choice"] = kwargs["tool_choice"]
        if (
            "parallel_tool_calls" in kwargs
            and kwargs["parallel_tool_calls"] is not None
        ):
            payload["parallel_tool_calls"] = kwargs["parallel_tool_calls"]

        # Map reasoning_effort to model-specific thinking config
        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort is not None:
            thinking_config = _map_reasoning_effort_to_config(reasoning_effort, model)
            payload.update(thinking_config)

        # Structured output
        if "response_format" in kwargs and kwargs["response_format"] is not None:
            payload["text"] = {"format": kwargs["response_format"]}

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

                created = int(
                    datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
                )
            except (ValueError, TypeError):
                created = int(time.time())

        response_id = data.get("id", f"resp-{uuid.uuid4().hex[:8]}")

        # Extract output text and reasoning from the response
        output_text = data.get("output_text", "")
        reasoning_content = ""

        # If output_text is not directly available, extract from output array
        if "output" in data:
            for item in data.get("output", []):
                if item.get("type") == "message" and item.get("role") == "assistant":
                    for content_part in item.get("content", []):
                        if (
                            content_part.get("type") == "output_text"
                            and not output_text
                        ):
                            output_text = content_part.get("text", "")
                        elif content_part.get("type") == "reasoning":
                            reasoning_content += content_part.get("text", "")
                elif item.get("type") == "reasoning":
                    summary = item.get("summary")
                    if isinstance(summary, list):
                        for part in summary:
                            text = part.get("text")
                            if text:
                                reasoning_content += text
                    else:
                        text = item.get("text")
                        if text:
                            reasoning_content += text

        # Build choices in Chat Completions format
        choices = [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output_text,
                    **(
                        {"reasoning_content": reasoning_content}
                        if reasoning_content
                        else {}
                    ),
                },
                "finish_reason": data.get("status", "completed") == "completed"
                and "stop"
                or "stop",
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
                completion_tokens=usage.get(
                    "output_tokens", usage.get("completion_tokens", 0)
                ),
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

                if event_type == "response.output_item.done":
                    item = evt.get("item", {})
                    if item.get("type") == "reasoning":
                        lib_logger.debug(
                            "[GitHubCopilot] reasoning item event: %s",
                            json.dumps(item, default=str)[:500],
                        )

                if event_type == "response.content_part.delta":
                    content_part = evt.get("part", {})
                    if content_part.get("type") == "reasoning":
                        lib_logger.debug(
                            "[GitHubCopilot] reasoning part delta: %s",
                            str(content_part.get("text", ""))[:500],
                        )

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

                # Handle response completion events
                elif event_type in ("response.done", "response.completed"):
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
                    break  # Exit stream loop - response is complete

                elif event_type == "response.output_item.done":
                    item = evt.get("item", {})
                    item_type = item.get("type")
                    if item_type in ("function_call", "tool_call"):
                        call_id = item.get("call_id") or item.get("id")
                        call_index = item.get("index", 0)
                        name = item.get("name") or item.get("function", {}).get("name")
                        arguments = item.get("arguments") or item.get(
                            "function", {}
                        ).get("arguments", "")
                        if name:
                            tool_call = {
                                "index": call_index,
                                "id": call_id or f"call_{uuid.uuid4().hex[:8]}",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": arguments or "",
                                },
                            }
                            chunk = litellm.ModelResponse(
                                id=response_id,
                                created=created,
                                model=f"github_copilot/{model}",
                                object="chat.completion.chunk",
                                choices=[
                                    {
                                        "index": 0,
                                        "delta": {"tool_calls": [tool_call]},
                                        "finish_reason": None,
                                    }
                                ],
                            )
                            yield chunk
                    elif item_type == "reasoning":
                        reasoning_text = ""
                        summary = item.get("summary")
                        if isinstance(summary, list):
                            for part in summary:
                                text = part.get("text")
                                if text:
                                    reasoning_text += text
                        else:
                            text = item.get("text")
                            if text:
                                reasoning_text += text
                        if reasoning_text:
                            chunk = litellm.ModelResponse(
                                id=response_id,
                                created=created,
                                model=f"github_copilot/{model}",
                                object="chat.completion.chunk",
                                choices=[
                                    {
                                        "index": 0,
                                        "delta": {
                                            "reasoning_content": reasoning_text,
                                            "role": "assistant",
                                        },
                                        "finish_reason": None,
                                    }
                                ],
                            )
                            yield chunk
                    continue

                # Skip events that contain complete text already streamed via deltas
                elif event_type in (
                    "response.output_text.done",
                    "response.content_part.done",
                ):
                    continue

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
                    elif content_part.get("type") == "reasoning":
                        reasoning_text = content_part.get("text", "")
                        if reasoning_text:
                            chunk = litellm.ModelResponse(
                                id=response_id,
                                created=created,
                                model=f"github_copilot/{model}",
                                object="chat.completion.chunk",
                                choices=[
                                    {
                                        "index": 0,
                                        "delta": {
                                            "reasoning_content": reasoning_text,
                                            "role": "assistant",
                                        },
                                        "finish_reason": None,
                                    }
                                ],
                            )
                            yield chunk
