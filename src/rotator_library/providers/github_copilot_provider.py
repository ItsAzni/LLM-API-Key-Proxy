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

import asyncio
import json
import logging
import os
import random
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, TYPE_CHECKING

import httpx
import litellm

from ..timeout_config import TimeoutConfig
from ..anthropic_compat.translator import anthropic_to_openai_messages

from .provider_interface import ProviderInterface
from .github_copilot_auth_base import GitHubCopilotAuthBase

if TYPE_CHECKING:
    from ..usage.manager import UsageManager

lib_logger = logging.getLogger("rotator_library")

# =============================================================================
# API CONFIGURATION
# =============================================================================

# GitHub Copilot API base URLs
COPILOT_API_BASE = "https://api.githubcopilot.com"

# GitHub API base URL for quota fetching
GITHUB_API_BASE = "https://api.github.com"

# Concurrency limit for quota fetches
QUOTA_FETCH_CONCURRENCY = 5

# OpenCode version cache (fetched once on first use)
_opencode_version_cache: Optional[str] = None
_OPENCODE_FALLBACK_VERSION = "1.1.36"  # Fallback if GitHub fetch fails

# =============================================================================
# MESSAGES API 24-HOUR RETRY PREVENTION
# =============================================================================
# When Messages API fails for a model, we record the failure timestamp and
# don't retry for 24 hours. This matches OpenCode's approach of not using
# Messages API due to rate limiting concerns.

# Cache for Messages API failures: {model: failure_timestamp}
_messages_api_failure_cache: Dict[str, float] = {}

# 24-hour cooldown in seconds (configurable via env var)
MESSAGES_API_COOLDOWN_SECONDS = int(
    os.getenv("GITHUB_COPILOT_MESSAGES_API_COOLDOWN", str(24 * 60 * 60))
)


def _is_messages_api_on_cooldown(model: str) -> bool:
    """Check if Messages API is on 24-hour cooldown for this model."""
    failure_time = _messages_api_failure_cache.get(model)
    if failure_time is None:
        return False

    time_since_failure = time.time() - failure_time
    if time_since_failure < MESSAGES_API_COOLDOWN_SECONDS:
        hours_remaining = (MESSAGES_API_COOLDOWN_SECONDS - time_since_failure) / 3600
        lib_logger.debug(
            "[GitHubCopilot] Messages API on cooldown for %s (%.1f hours remaining)",
            model,
            hours_remaining,
        )
        return True

    # Cooldown expired, clear the cache entry
    del _messages_api_failure_cache[model]
    lib_logger.info(
        "[GitHubCopilot] Messages API cooldown expired for %s, will try again",
        model,
    )
    return False


def _record_messages_api_failure(model: str) -> None:
    """Record that Messages API failed for this model, starting 24-hour cooldown."""
    _messages_api_failure_cache[model] = time.time()
    cooldown_hours = MESSAGES_API_COOLDOWN_SECONDS / 3600
    lib_logger.warning(
        "[GitHubCopilot] Messages API failed for %s - will not retry for %.0f hours",
        model,
        cooldown_hours,
    )


def _env_truthy(name: str, default: str = "true") -> bool:
    value = os.getenv(name, default)
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _should_try_messages_api() -> bool:
    """
    Check if Messages API is enabled via env flag for Claude models.

    OpenCode uses Chat Completions for all models, including Claude.
    Set GITHUB_COPILOT_TRY_MESSAGES_API=true to try Messages API first.

    Default: false (use Chat Completions like OpenCode)
    """
    return _env_truthy("GITHUB_COPILOT_TRY_MESSAGES_API", default="false")


def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _tool_choice_forces_use(tool_choice: Any) -> bool:
    """Check if tool_choice forces the model to use a specific tool."""
    if tool_choice is None:
        return False
    if isinstance(tool_choice, str):
        return tool_choice.strip().lower() in ("required", "any", "force", "tool")
    if isinstance(tool_choice, dict):
        tc_type = tool_choice.get("type")
        return tc_type in ("function", "tool")
    return False


# Quota refresh interval in seconds (default: 5 minutes)
# Defined after _env_int so we can use it
COPILOT_QUOTA_REFRESH_INTERVAL = _env_int("GITHUB_COPILOT_QUOTA_REFRESH_INTERVAL", 300)


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

# Anthropic compatibility (used for Copilot Claude models)
ANTHROPIC_VERSION = "2023-06-01"
# OpenCode Copilot parity: only set the interleaved thinking beta.
COPILOT_ANTHROPIC_BETA = "interleaved-thinking-2025-05-14"

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
    "claude-sonnet-4.6",
    "claude-opus-4",
    "claude-opus-4.5",
    "claude-opus-4.6",
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

# =============================================================================
# MODEL NAME ALIASES
# =============================================================================
# Maps alternative model names (e.g. Anthropic dash-style versioning) to the
# canonical names accepted by GitHub Copilot.
#
# Claude Code and the Anthropic SDK use dashes for version separators
# (e.g. claude-opus-4-6), whereas GitHub Copilot uses dots (claude-opus-4.6).
# This mapping transparently rewrites the model name before the API call.
MODEL_ALIASES: Dict[str, str] = {
    # Claude 4.5 aliases (Anthropic dash-style → Copilot dot-style)
    "claude-sonnet-4-5": "claude-sonnet-4.5",
    "claude-opus-4-5": "claude-opus-4.5",
    "claude-haiku-4-5": "claude-haiku-4.5",
    # Claude 4.6 aliases
    "claude-sonnet-4-6": "claude-sonnet-4.6",
    "claude-opus-4-6": "claude-opus-4.6",
    # Claude 3.x aliases
    "claude-3-5-sonnet": "claude-3.5-sonnet",
    "claude-3-7-sonnet": "claude-3.7-sonnet",
    "claude-3-7-sonnet-thought": "claude-3.7-sonnet-thought",
    # GPT version aliases
    "gpt-4-1": "gpt-4.1",
    "gpt-5-1": "gpt-5.1",
    "gpt-5-2": "gpt-5.2",
    "gpt-5-1-codex": "gpt-5.1-codex",
    "gpt-5-1-codex-mini": "gpt-5.1-codex-mini",
    "gpt-5-1-codex-max": "gpt-5.1-codex-max",
}


def _normalize_model_name(model: str) -> str:
    """Normalize a model name using the alias table.

    If the model (after stripping provider prefix) matches an alias key,
    the canonical name is returned.  Otherwise the input is returned unchanged.
    """
    clean = model.split("/")[-1] if "/" in model else model
    return MODEL_ALIASES.get(clean, clean)


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


def _claude_supports_thinking(model: str) -> bool:
    """Check if a Claude model supports extended thinking.

    Only Sonnet 3.7+, Sonnet 4+, and Opus models support thinking.
    Haiku and older models do not.
    """
    clean = model.split("/")[-1].lower() if "/" in model else model.lower()
    if "haiku" in clean:
        return False
    if "3.5" in clean:
        return False
    return True


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
            # Copilot Responses API rejects "minimal" for some GPT-5 models.
            # Map to the lowest supported tier to avoid 400s.
            lib_logger.debug(
                "[GitHubCopilot] reasoning_effort 'minimal' mapped to 'low' for model %s",
                clean_model,
            )
            effort = "low"
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
            "xhigh": 32000,  # Map xhigh to highest supported
            "max": 32000,  # Map max to highest supported
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
                "xhigh": 24576,  # Map xhigh to highest supported
                "max": 24576,  # Map max to highest supported
            }
            return {"thinkingConfig": {"thinkingBudget": budgets.get(effort, -1)}}

    return {}


def _map_reasoning_effort_to_anthropic_thinking(
    reasoning_effort: Optional[str],
    model: str,
) -> Dict[str, Any]:
    """Map reasoning_effort to Anthropic Messages API thinking config."""
    if not _is_claude_model(model):
        return {}

    effort_raw = "auto" if reasoning_effort is None else str(reasoning_effort).strip()
    effort = effort_raw.lower() if effort_raw else "auto"
    if effort in ("disable", "off", "none"):
        return {"thinking": {"type": "disabled"}}

    base = _map_reasoning_effort_to_config(effort, model)
    budget = None
    if isinstance(base, dict):
        thinking = base.get("thinking")
        if isinstance(thinking, dict):
            budget = thinking.get("budget_tokens")

    if isinstance(budget, int) and budget > 0:
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}

    # Default to enabling thinking when using interleaved-thinking.
    return {"thinking": {"type": "enabled", "budget_tokens": 16000}}


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

    def _sanitize_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Sanitize messages to ensure tool calls have required fields.

        GitHub Copilot API requires tool calls to have both 'id' and
        'function.name'. This method filters out invalid tool calls.

        Args:
            messages: List of message dictionaries

        Returns:
            Sanitized list of messages
        """
        sanitized = []
        for msg in messages:
            msg_copy = dict(msg)
            role = msg_copy.get("role", "")

            # Handle assistant messages with tool_calls
            if role == "assistant" and "tool_calls" in msg_copy:
                tool_calls = msg_copy.get("tool_calls", [])
                if tool_calls:
                    valid_tool_calls = []
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        # Check for required fields
                        tc_id = tc.get("id")
                        func = tc.get("function", {})
                        func_name = func.get("name") if isinstance(func, dict) else None

                        # Skip tool calls missing id or function name
                        if not tc_id or not func_name:
                            lib_logger.warning(
                                f"[GitHubCopilot] Skipping invalid tool call: "
                                f"id={tc_id!r}, name={func_name!r}"
                            )
                            continue
                        valid_tool_calls.append(tc)

                    if valid_tool_calls:
                        msg_copy["tool_calls"] = valid_tool_calls
                    else:
                        # Remove tool_calls if none are valid
                        del msg_copy["tool_calls"]

            sanitized.append(msg_copy)
        return sanitized

    def _transform_messages_for_reasoning(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Transform messages to include reasoning fields for multi-turn reasoning.

        OpenCode parity: Only include reasoning_text when reasoning_opaque exists.
        This ensures the model continues from its previous reasoning state rather
        than restarting reasoning from scratch each turn.

        Args:
            messages: List of message dictionaries

        Returns:
            Transformed list of messages with reasoning fields preserved
        """
        transformed = []
        for msg in messages:
            msg_copy = dict(msg)
            role = msg_copy.get("role", "")

            if role == "assistant":
                # OpenCode parity: Only include reasoning_text when reasoning_opaque exists
                # This preserves multi-turn reasoning continuity
                reasoning_opaque = msg_copy.get("reasoning_opaque")
                if reasoning_opaque:
                    # Get reasoning_text (or reasoning_content for compatibility)
                    reasoning = msg_copy.get("reasoning_text") or msg_copy.get(
                        "reasoning_content"
                    )
                    if reasoning:
                        msg_copy["reasoning_text"] = reasoning
                    # reasoning_opaque is already in msg_copy, preserve it
                else:
                    # No opaque token - remove reasoning_text to force fresh reasoning
                    msg_copy.pop("reasoning_text", None)
                    msg_copy.pop("reasoning_content", None)

            transformed.append(msg_copy)
        return transformed

    def _transform_user_messages_to_tool_results(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Transform user messages into synthetic tool call + result pairs.

        This eliminates 'role: user' from the context, making the entire
        conversation appear agent-driven (as if using AskUserQuestion tool
        to gather user input).

        When enabled:
        - User messages become: assistant tool_call + tool result pairs
        - x-initiator naturally becomes "agent" since last.role != "user"
        - The conversation appears to be an agent using tools to gather input

        Enabled via: GITHUB_COPILOT_SYNTHETIC_USER_MESSAGES=true
        Probability: GITHUB_COPILOT_SYNTHETIC_RATIO (0-1, default 1.0)

        Args:
            messages: List of message dictionaries

        Returns:
            Transformed messages with user messages converted to tool results
        """
        if not _env_truthy("GITHUB_COPILOT_SYNTHETIC_USER_MESSAGES", "false"):
            return messages

        ratio = 1.0
        try:
            ratio = float(os.getenv("GITHUB_COPILOT_SYNTHETIC_RATIO", "1.0"))
        except (TypeError, ValueError):
            ratio = 1.0

        transformed: List[Dict[str, Any]] = []
        user_msgs_transformed = 0
        user_msgs_kept = 0

        for msg in messages:
            role = msg.get("role", "")

            if role == "user":
                # Apply transformation probability to ALL messages (including first)
                if random.random() > ratio:
                    transformed.append(msg)
                    user_msgs_kept += 1
                    continue

                # Generate unique call ID
                call_id = f"q_{uuid.uuid4().hex[:12]}"
                content = msg.get("content", "")

                # Create synthetic assistant message with tool call
                transformed.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": "_relay_user_input",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    }
                )

                # Create tool result with user's actual content
                tool_content = (
                    content if isinstance(content, str) else json.dumps(content)
                )
                transformed.append(
                    {"role": "tool", "tool_call_id": call_id, "content": tool_content}
                )
                user_msgs_transformed += 1
            else:
                transformed.append(msg)

        if user_msgs_transformed > 0 or user_msgs_kept > 0:
            lib_logger.info(
                "[GitHubCopilot] Synthetic user messages: %d transformed, %d kept (ratio=%.2f)",
                user_msgs_transformed,
                user_msgs_kept,
                ratio,
            )

        # Ensure the first non-system message is role "user".
        # The Copilot backend converts to Anthropic format for Claude,
        # which requires conversations to start with a user message.
        # When all user messages are transformed, the first non-system
        # message becomes "assistant" (tool_use), causing the backend
        # to reject the tool_result for lacking a prior tool_use.
        if user_msgs_transformed > 0:
            first_non_system = next(
                (
                    i
                    for i, m in enumerate(transformed)
                    if m.get("role") not in ("system", "developer")
                ),
                None,
            )
            if (
                first_non_system is not None
                and transformed[first_non_system].get("role") != "user"
            ):
                transformed.insert(
                    first_non_system,
                    {
                        "role": "user",
                        "content": "Respond to the user's message provided via the tool result below.",
                    },
                )

        return transformed

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
        Detect if a request should be marked as agent-initiated.

        OpenCode parity: x-initiator=agent iff last.role != "user"

        When synthetic user messages are enabled, all user messages become
        tool results, so last.role will never be "user" and all requests
        become agent-initiated naturally.

        Args:
            messages: List of message dictionaries

        Returns:
            True if request should use x-initiator: "agent"
        """
        if not messages:
            return False

        # OpenCode behavior: x-initiator=agent iff last.role != "user"
        last_msg = messages[-1]
        last_role = last_msg.get("role", "user")
        return last_role != "user"

    async def _build_copilot_headers(
        self,
        credential_path: str,
        is_vision: bool = False,
        is_agent: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Build headers for GitHub Copilot API requests.

        Args:
            credential_path: Path to the credential file
            is_vision: True if request contains vision content
            is_agent: True if request is agent-initiated
            extra_headers: Additional headers to include

        Returns:
            Dictionary of headers for the API request
        """
        auth_header = await self.get_auth_header(credential_path)

        # Match OpenCode's header behavior/order:
        #   x-initiator -> init.headers -> User-Agent -> Authorization -> Openai-Intent
        init_headers: Dict[str, str] = {}
        if isinstance(extra_headers, dict):
            init_headers = {str(k): str(v) for k, v in extra_headers.items()}
            # Match fetch/Headers behavior from OpenCode where auth-like keys are
            # effectively case-insensitive and stripped before request dispatch.
            init_headers = {
                k: v
                for k, v in init_headers.items()
                if k.lower() not in ("x-api-key", "authorization")
            }

        initiator = "agent" if is_agent else "user"

        headers: Dict[str, str] = {
            "x-initiator": initiator,
            **init_headers,
            "User-Agent": USER_AGENT,
            **auth_header,
            "Openai-Intent": "conversation-edits",
        }

        if is_vision:
            headers["Copilot-Vision-Request"] = "true"

        # Remove conflicting auth headers (OpenCode parity)
        headers.pop("x-api-key", None)
        headers.pop("authorization", None)

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
        extra_headers = kwargs.pop("extra_headers", None)
        # Never forward temperature for GitHub Copilot requests.
        kwargs.pop("temperature", None)

        # Normalize model name (strip provider prefix and apply aliases)
        if "/" in model:
            model = model.split("/", 1)[1]
        model = _normalize_model_name(model)

        # =====================================================================
        # EARLY MESSAGE TRANSFORMATION
        # Transform messages BEFORE header detection so x-initiator is correct
        # =====================================================================
        # Sanitize messages to remove invalid tool calls
        transformed_messages = self._sanitize_messages(messages)
        # Transform user messages to synthetic tool results (agent spoofing)
        transformed_messages = self._transform_user_messages_to_tool_results(
            transformed_messages
        )
        # Apply reasoning field transformation for OpenCode parity
        transformed_messages = self._transform_messages_for_reasoning(
            transformed_messages
        )

        # Detect content types using TRANSFORMED messages
        is_vision = self._detect_vision_content(transformed_messages)
        is_agent = self._detect_agent_initiated(transformed_messages)

        lib_logger.debug(
            "[GitHubCopilot] x-initiator will be '%s' (is_agent=%s, last_role=%s)",
            "agent" if is_agent else "user",
            is_agent,
            transformed_messages[-1].get("role", "unknown")
            if transformed_messages
            else "empty",
        )

        # Get API base URL
        api_base = self._get_api_base(credential_path)

        # Build headers
        headers = await self._build_copilot_headers(
            credential_path,
            is_vision=is_vision,
            is_agent=is_agent,
            extra_headers=extra_headers,
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
                messages=transformed_messages,  # Use transformed messages
                stream=stream,
                **filtered_kwargs,
            )

        # Claude models: optionally try Anthropic Messages API for thinking support.
        # By default (OpenCode parity), use Chat Completions directly.
        # Set GITHUB_COPILOT_TRY_MESSAGES_API=true to try Messages API first.
        # Messages API has 24-hour cooldown after failure to avoid repeated failures.
        if (
            _is_claude_model(model)
            and _should_try_messages_api()
            and not _is_messages_api_on_cooldown(model)
        ):
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
            if not stream:
                # Non-streaming: simple try/except
                try:
                    return await self._anthropic_messages_completion(
                        client=client,
                        api_base=api_base,
                        headers=headers,
                        model=model,
                        messages=transformed_messages,  # Use transformed messages
                        stream=False,
                        **filtered_kwargs,
                    )
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        # Record failure to prevent retries for 24 hours
                        _record_messages_api_failure(model)
                        lib_logger.warning(
                            "[GitHubCopilot] Messages API returned 404 for %s, falling back to Chat Completions (thinking disabled)",
                            model,
                        )
                        # Fall through to Chat Completions below
                    else:
                        raise
            else:
                # Streaming: wrap in fallback generator.
                # Try Messages API first; if 404, fall back to Chat Completions.
                return self._stream_claude_with_fallback(
                    client=client,
                    api_base=api_base,
                    headers=headers,
                    model=model,
                    messages=transformed_messages,  # Use transformed messages
                    filtered_kwargs=filtered_kwargs,
                    fallback_kwargs=kwargs,
                )

        # Build request payload for Chat Completions API
        # Messages are already transformed at the top of the function
        payload: Dict[str, Any] = {
            "model": model,
            "messages": transformed_messages,
            "store": False,
        }
        if stream:
            payload["stream"] = True

        # Copy over optional parameters
        optional_params = [
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
            # OpenCode parity parameters
            "verbosity",  # Controls text verbosity
        ]
        for param in optional_params:
            if param in kwargs and kwargs[param] is not None:
                payload[param] = kwargs[param]

        # Handle reasoning parameters based on model type
        # For Claude models: thinking_budget (direct passthrough or mapped from reasoning_effort)
        # For GPT/o-series: reasoning.effort + reasoning.summary
        reasoning_effort = kwargs.get("reasoning_effort")
        thinking_budget = kwargs.get("thinking_budget")
        reasoning_summary = kwargs.get("reasoning_summary") or kwargs.get(
            "reasoningSummary"
        )

        if _is_claude_model(model) and _claude_supports_thinking(model):
            # Check if tool_choice forces use - if so, skip thinking
            # (Anthropic API: "Thinking may not be enabled when tool_choice forces tool use")
            tools = kwargs.get("tools") or payload.get("tools")
            tool_choice = kwargs.get("tool_choice") or payload.get("tool_choice")
            if tools and _tool_choice_forces_use(tool_choice):
                lib_logger.warning(
                    "[GitHubCopilot] Disabling thinking because tool_choice forces tool use for model %s",
                    model,
                )
            else:
                # Claude: use thinking_budget parameter
                final_thinking_budget = None
                if thinking_budget is not None:
                    # Direct passthrough takes priority
                    final_thinking_budget = thinking_budget
                elif reasoning_effort is not None:
                    # Map reasoning_effort to thinking_budget for Claude
                    thinking_config = _map_reasoning_effort_to_config(
                        reasoning_effort, model
                    )
                    budget_tokens = thinking_config.get("thinking", {}).get(
                        "budget_tokens"
                    )
                    if budget_tokens:
                        final_thinking_budget = budget_tokens

                if final_thinking_budget is not None:
                    # OpenCode parity: use flat thinking_budget at top level
                    # See: opencode/packages/opencode/src/provider/sdk/copilot/chat/openai-compatible-chat-language-model.ts
                    payload["thinking_budget"] = final_thinking_budget
                    # Ensure max_tokens > thinking_budget (Anthropic requirement)
                    current_max = payload.get("max_tokens") or 0
                    if current_max <= final_thinking_budget:
                        payload["max_tokens"] = final_thinking_budget + 4096
                    lib_logger.info(
                        "[GitHubCopilot] Final request thinking_budget for %s: %s, max_tokens: %s%s",
                        model,
                        final_thinking_budget,
                        payload["max_tokens"],
                        f" (from reasoning_effort={reasoning_effort})"
                        if reasoning_effort and not thinking_budget
                        else "",
                    )
        elif _is_claude_model(model):
            # Claude models that don't support thinking (e.g. haiku, 3.5)
            lib_logger.debug(
                "[GitHubCopilot] Skipping thinking_budget for %s (model does not support extended thinking)",
                model,
            )
        elif _is_gpt5_or_o_series(model):
            # GPT-5/o-series: use reasoning.effort + reasoning.summary
            if reasoning_effort is not None:
                thinking_config = _map_reasoning_effort_to_config(
                    reasoning_effort, model
                )
                payload.update(thinking_config)
                lib_logger.debug(
                    "[GitHubCopilot] reasoning_effort=%s mapped_config=%s",
                    reasoning_effort,
                    json.dumps(thinking_config, default=str)[:500],
                )
            # Add reasoningSummary for GPT models (OpenCode parity)
            if reasoning_summary is not None:
                if "reasoning" in payload:
                    payload["reasoning"]["summary"] = reasoning_summary
                else:
                    payload["reasoning"] = {"summary": reasoning_summary}
        elif reasoning_effort is not None:
            # Other models: use generic mapping
            thinking_config = _map_reasoning_effort_to_config(reasoning_effort, model)
            payload.update(thinking_config)
            lib_logger.debug(
                "[GitHubCopilot] reasoning_effort=%s mapped_config=%s",
                reasoning_effort,
                json.dumps(thinking_config, default=str)[:500],
            )

        # OpenCode parity: apply copilot_cache_control hints to system and recent messages
        # See: opencode transform.ts:191-193
        msgs = payload.get("messages", [])
        if msgs:
            # First 2 system messages get cache control
            system_count = 0
            for msg in msgs:
                if msg.get("role") == "system" and system_count < 2:
                    msg["copilot_cache_control"] = {"type": "ephemeral"}
                    system_count += 1
            # Last 2 non-system messages get cache control
            non_system_indices = [
                i for i, msg in enumerate(msgs) if msg.get("role") != "system"
            ]
            for idx in non_system_indices[-2:]:
                msgs[idx]["copilot_cache_control"] = {"type": "ephemeral"}

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

    async def _stream_claude_with_fallback(
        self,
        client: httpx.AsyncClient,
        api_base: str,
        headers: Dict[str, str],
        model: str,
        messages: List[Dict[str, Any]],
        filtered_kwargs: Dict[str, Any],
        fallback_kwargs: Dict[str, Any],
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """
        Stream Claude via Messages API, falling back to Chat Completions on 404.

        This wraps the Messages API stream and catches 404 errors. If the
        Messages API returns 404, it falls back to Chat Completions (without
        thinking support).
        """
        try:
            messages_gen = self._stream_anthropic_messages_with_error_check(
                client=client,
                api_base=api_base,
                headers=headers,
                model=model,
                messages=messages,
                **filtered_kwargs,
            )
            async for chunk in messages_gen:
                yield chunk
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # Record failure to prevent retries for 24 hours
                _record_messages_api_failure(model)
                lib_logger.warning(
                    "[GitHubCopilot] Messages API returned 404 for %s, falling back to Chat Completions (thinking disabled)",
                    model,
                )
                # Filter out params that are passed explicitly to avoid conflicts
                safe_kwargs = {
                    k: v
                    for k, v in fallback_kwargs.items()
                    if k
                    not in (
                        "model",
                        "messages",
                        "stream",
                        "credential_identifier",
                        "credential_path",
                    )
                }
                # Fall back to Chat Completions
                chat_gen = await self._execute_chat_completions_stream(
                    client=client,
                    api_base=api_base,
                    headers=headers,
                    model=model,
                    messages=messages,
                    **safe_kwargs,
                )
                async for chunk in chat_gen:
                    yield chunk
            else:
                raise

    async def _stream_anthropic_messages_with_error_check(
        self,
        client: httpx.AsyncClient,
        api_base: str,
        headers: Dict[str, str],
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """
        Wrapper that eagerly checks for HTTP errors before streaming.

        The inner generator (_stream_anthropic_messages) raises HTTPStatusError
        inside the generator, which makes it hard to catch at the caller level.
        This wrapper makes the initial request, checks for errors, and then
        yields from the inner generator.
        """
        gen = await self._anthropic_messages_completion(
            client=client,
            api_base=api_base,
            headers=headers,
            model=model,
            messages=messages,
            stream=True,
            **kwargs,
        )
        # Yield from the generator - errors will propagate
        async for chunk in gen:
            yield chunk

    async def _execute_chat_completions_stream(
        self,
        client: httpx.AsyncClient,
        api_base: str,
        headers: Dict[str, str],
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """Execute Chat Completions API for streaming fallback.

        Note: Messages are expected to be already transformed (sanitized,
        synthetic user messages applied, reasoning fields processed) before
        being passed to this function.
        """
        # Build request payload for Chat Completions API
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,  # Already transformed
            "stream": True,
        }

        # Copy over optional parameters
        optional_params = [
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
            # OpenCode parity parameters
            "verbosity",  # Controls text verbosity
        ]
        for param in optional_params:
            if param in kwargs and kwargs[param] is not None:
                payload[param] = kwargs[param]

        # Handle reasoning parameters based on model type
        # For Claude models: thinking_budget (direct passthrough or mapped from reasoning_effort)
        # For GPT/o-series: reasoning.effort + reasoning.summary
        reasoning_effort = kwargs.get("reasoning_effort")
        thinking_budget = kwargs.get("thinking_budget")
        reasoning_summary = kwargs.get("reasoning_summary") or kwargs.get(
            "reasoningSummary"
        )

        if _is_claude_model(model) and _claude_supports_thinking(model):
            # Check if tool_choice forces use - if so, skip thinking
            # (Anthropic API: "Thinking may not be enabled when tool_choice forces tool use")
            tools = kwargs.get("tools") or payload.get("tools")
            tool_choice = kwargs.get("tool_choice") or payload.get("tool_choice")
            if tools and _tool_choice_forces_use(tool_choice):
                lib_logger.warning(
                    "[GitHubCopilot] Disabling thinking because tool_choice forces tool use for model %s",
                    model,
                )
            else:
                # Claude: use thinking_budget parameter
                final_thinking_budget = None
                if thinking_budget is not None:
                    # Direct passthrough takes priority
                    final_thinking_budget = thinking_budget
                elif reasoning_effort is not None:
                    # Map reasoning_effort to thinking_budget for Claude
                    thinking_config = _map_reasoning_effort_to_config(
                        reasoning_effort, model
                    )
                    budget_tokens = thinking_config.get("thinking", {}).get(
                        "budget_tokens"
                    )
                    if budget_tokens:
                        final_thinking_budget = budget_tokens

                if final_thinking_budget is not None:
                    # OpenCode parity: use flat thinking_budget at top level
                    # See: opencode/packages/opencode/src/provider/sdk/copilot/chat/openai-compatible-chat-language-model.ts
                    payload["thinking_budget"] = final_thinking_budget
                    # Ensure max_tokens > thinking_budget (Anthropic requirement)
                    current_max = payload.get("max_tokens") or 0
                    if current_max <= final_thinking_budget:
                        payload["max_tokens"] = final_thinking_budget + 4096
                    lib_logger.info(
                        "[GitHubCopilot] Final request thinking_budget for %s: %s, max_tokens: %s%s",
                        model,
                        final_thinking_budget,
                        payload["max_tokens"],
                        f" (from reasoning_effort={reasoning_effort})"
                        if reasoning_effort and not thinking_budget
                        else "",
                    )
        elif _is_claude_model(model):
            # Claude models that don't support thinking (e.g. haiku, 3.5)
            lib_logger.debug(
                "[GitHubCopilot] Skipping thinking_budget for %s (model does not support extended thinking)",
                model,
            )
        elif _is_gpt5_or_o_series(model):
            # GPT-5/o-series: use reasoning.effort + reasoning.summary
            if reasoning_effort is not None:
                thinking_config = _map_reasoning_effort_to_config(
                    reasoning_effort, model
                )
                payload.update(thinking_config)
                lib_logger.debug(
                    "[GitHubCopilot] reasoning_effort=%s mapped_config=%s",
                    reasoning_effort,
                    json.dumps(thinking_config, default=str)[:500],
                )
            # Add reasoningSummary for GPT models (OpenCode parity)
            if reasoning_summary is not None:
                if "reasoning" in payload:
                    payload["reasoning"]["summary"] = reasoning_summary
                else:
                    payload["reasoning"] = {"summary": reasoning_summary}
        elif reasoning_effort is not None:
            # Other models: use generic mapping
            thinking_config = _map_reasoning_effort_to_config(reasoning_effort, model)
            payload.update(thinking_config)
            lib_logger.debug(
                "[GitHubCopilot] reasoning_effort=%s mapped_config=%s",
                reasoning_effort,
                json.dumps(thinking_config, default=str)[:500],
            )

        endpoint = f"{api_base}/chat/completions"

        lib_logger.debug(
            f"Copilot fallback request to {model}: {json.dumps(payload, default=str)[:500]}..."
        )

        return self._stream_chat_response(client, endpoint, headers, payload, model)

    # =========================================================================
    # COPILOT CLAUDE (ANTHROPIC MESSAGES API) IMPLEMENTATION
    # =========================================================================

    def _openai_tools_to_anthropic(
        self, tools: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        if not tools:
            return []
        out: List[Dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") != "function":
                continue
            func = tool.get("function") or {}
            if not isinstance(func, dict):
                continue
            name = func.get("name")
            if not name:
                continue
            out.append(
                {
                    "name": name,
                    "description": func.get("description") or "",
                    "input_schema": func.get("parameters")
                    or {"type": "object", "properties": {}},
                }
            )
        return out

    def _openai_tool_choice_to_anthropic(
        self, tool_choice: Any
    ) -> Optional[Dict[str, Any]]:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            tc = tool_choice.strip().lower()
            if tc == "auto":
                return {"type": "auto"}
            if tc in ("required", "any"):
                return {"type": "any"}
            if tc == "none":
                return {"type": "none"}
            return {"type": "auto"}
        if isinstance(tool_choice, dict):
            # OpenAI: {"type":"function","function":{"name":"..."}}
            if tool_choice.get("type") == "function":
                func = tool_choice.get("function") or {}
                if isinstance(func, dict) and func.get("name"):
                    return {"type": "tool", "name": func["name"]}
        return {"type": "auto"}

    def _openai_messages_to_anthropic(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        system_parts: List[str] = []
        anthropic_messages: List[Dict[str, Any]] = []

        def _ensure_user_message() -> Dict[str, Any]:
            if anthropic_messages and anthropic_messages[-1].get("role") == "user":
                return anthropic_messages[-1]
            msg = {"role": "user", "content": []}
            anthropic_messages.append(msg)
            return msg

        def _append_message(role: str, blocks: List[Dict[str, Any]]) -> None:
            if not blocks:
                return
            if anthropic_messages and anthropic_messages[-1].get("role") == role:
                anthropic_messages[-1]["content"].extend(blocks)
                return
            anthropic_messages.append({"role": role, "content": blocks})

        for msg in messages or []:
            if not isinstance(msg, dict):
                continue
            role = (msg.get("role") or "user").lower()

            if role in ("system", "developer"):
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    system_parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "")
                            if text and text.strip():
                                system_parts.append(text)
                continue

            if role == "tool":
                tool_call_id = msg.get("tool_call_id") or ""
                tool_content = msg.get("content")
                if isinstance(tool_content, str):
                    content_str = tool_content
                elif tool_content is None:
                    content_str = ""
                else:
                    content_str = json.dumps(tool_content)
                user_msg = _ensure_user_message()
                user_msg["content"].append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": content_str,
                    }
                )
                continue

            content = msg.get("content")
            blocks: List[Dict[str, Any]] = []

            if isinstance(content, str):
                if content:
                    blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type")
                    if ptype == "text":
                        blocks.append({"type": "text", "text": part.get("text", "")})
                    elif ptype == "image_url":
                        image_url = part.get("image_url")
                        url = ""
                        if isinstance(image_url, dict):
                            url = image_url.get("url", "")
                        elif isinstance(image_url, str):
                            url = image_url
                        if url:
                            blocks.append(
                                {
                                    "type": "image",
                                    "source": {"type": "url", "url": url},
                                }
                            )

            if role == "assistant":
                tool_calls = msg.get("tool_calls") or []
                for tc in tool_calls:
                    if not isinstance(tc, dict) or tc.get("type") != "function":
                        continue
                    call_id = tc.get("id") or f"toolu_{uuid.uuid4().hex[:12]}"
                    func = tc.get("function") or {}
                    if not isinstance(func, dict):
                        continue
                    name = func.get("name") or ""
                    args_raw = func.get("arguments") or "{}"
                    try:
                        input_data = (
                            json.loads(args_raw) if isinstance(args_raw, str) else {}
                        )
                    except json.JSONDecodeError:
                        input_data = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call_id,
                            "name": name,
                            "input": input_data,
                        }
                    )
                _append_message("assistant", blocks)
            else:
                _append_message("user", blocks)

        system = (
            "\n\n".join([p for p in system_parts if p and p.strip()]).strip() or None
        )
        return system, anthropic_messages

    def _map_anthropic_stop_reason_to_finish_reason(
        self, stop_reason: Optional[str]
    ) -> str:
        if not stop_reason:
            return "stop"
        sr = str(stop_reason)
        if sr == "end_turn":
            return "stop"
        if sr == "max_tokens":
            return "length"
        if sr == "tool_use":
            return "tool_calls"
        return "stop"

    async def _anthropic_messages_completion(
        self,
        client: httpx.AsyncClient,
        api_base: str,
        headers: Dict[str, str],
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool,
        **kwargs,
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        # Copilot expects Anthropic-style headers for Claude.
        h = dict(headers)
        h["anthropic-version"] = ANTHROPIC_VERSION
        h["anthropic-beta"] = COPILOT_ANTHROPIC_BETA
        if stream:
            h.setdefault("Accept", "text/event-stream")

        system, anthropic_messages = self._openai_messages_to_anthropic(messages)
        max_tokens = kwargs.get("max_tokens")
        try:
            max_tokens_int = int(max_tokens) if max_tokens is not None else 4096
        except (TypeError, ValueError):
            max_tokens_int = 4096

        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens_int,
            "messages": anthropic_messages,
        }
        if stream:
            payload["stream"] = True
        if system:
            payload["system"] = system

        if kwargs.get("top_p") is not None:
            payload["top_p"] = kwargs["top_p"]
        if kwargs.get("stop") is not None:
            stop = kwargs["stop"]
            if isinstance(stop, str):
                payload["stop_sequences"] = [stop]
            elif isinstance(stop, list):
                payload["stop_sequences"] = [s for s in stop if isinstance(s, str)]

        tools = kwargs.get("tools")
        if tools is not None:
            anthropic_tools = self._openai_tools_to_anthropic(tools)
            if anthropic_tools:
                payload["tools"] = anthropic_tools

        tool_choice = kwargs.get("tool_choice")
        tc = self._openai_tool_choice_to_anthropic(tool_choice)
        if tc:
            payload["tool_choice"] = tc

        reasoning_effort = kwargs.get("reasoning_effort")
        payload.update(
            _map_reasoning_effort_to_anthropic_thinking(reasoning_effort, model)
        )
        thinking = payload.get("thinking")
        if isinstance(thinking, dict) and thinking.get("type") == "enabled":
            payload["temperature"] = 1

        # Try /messages first (some Copilot versions), then /v1/messages
        endpoint = f"{api_base}/messages"
        lib_logger.debug(
            "[GitHubCopilot] Anthropic Messages request: model=%s endpoint=%s thinking=%s",
            model,
            endpoint,
            json.dumps(payload.get("thinking"), default=str),
        )

        if stream:
            return self._stream_anthropic_messages(client, endpoint, h, payload, model)
        return await self._non_stream_anthropic_messages(
            client, endpoint, h, payload, model
        )

    async def _non_stream_anthropic_messages(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
    ) -> litellm.ModelResponse:
        response = await client.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=TimeoutConfig.non_streaming(),
        )
        if response.status_code >= 400:
            error_text = response.text
            lib_logger.error(
                f"Copilot Anthropic Messages API error {response.status_code}: {error_text[:500]}"
            )
            raise httpx.HTTPStatusError(
                f"Copilot Anthropic Messages API error: {response.status_code}",
                request=response.request,
                response=response,
            )

        data = response.json()
        created = int(time.time())
        response_id = data.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}")

        content_blocks = data.get("content") or []
        openai_msgs = anthropic_to_openai_messages(
            [{"role": "assistant", "content": content_blocks}], system=None
        )
        assistant_msg = None
        for m in reversed(openai_msgs):
            if isinstance(m, dict) and m.get("role") == "assistant":
                assistant_msg = m
                break
        if not assistant_msg:
            assistant_msg = {"role": "assistant", "content": ""}

        finish_reason = self._map_anthropic_stop_reason_to_finish_reason(
            data.get("stop_reason")
        )

        response_obj = litellm.ModelResponse(
            id=response_id,
            created=created,
            model=f"github_copilot/{model}",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "message": assistant_msg,
                    "finish_reason": finish_reason,
                }
            ],
        )

        usage = data.get("usage") or {}
        if isinstance(usage, dict):
            prompt_tokens = usage.get("input_tokens", 0) or 0
            completion_tokens = usage.get("output_tokens", 0) or 0
            response_obj.usage = litellm.Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

        # Attach response headers for quota tracking
        response_obj._response_headers = {
            k.lower(): v for k, v in response.headers.items()
        }

        return response_obj

    # NOTE: GitHub Copilot's API does NOT expose a /messages/count_tokens
    # endpoint (returns 404).  We intentionally do NOT define a count_tokens()
    # method here so that RotatingClient.token_count_async() skips the native
    # call and falls back directly to the local LiteLLM tokenizer, which is
    # fast and reliable.

    async def _stream_anthropic_messages(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        created = int(time.time())
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        current_event: Optional[str] = None
        input_tokens = 0
        output_tokens = 0

        tool_calls: Dict[int, Dict[str, Any]] = {}
        tool_block_index_to_tc_index: Dict[int, int] = {}

        # Disable compression for streaming to get smooth token-by-token output.
        # Without this, gzip compression causes buffering until complete blocks
        # can be decompressed, resulting in chunky streaming.
        stream_headers = {**headers, "Accept-Encoding": "identity"}

        async with client.stream(
            "POST",
            endpoint,
            headers=stream_headers,
            json=payload,
            timeout=TimeoutConfig.streaming(),
        ) as response:
            if response.status_code >= 400:
                error_body = await response.aread()
                error_text = error_body.decode("utf-8", errors="ignore")
                lib_logger.error(
                    f"Copilot Anthropic Messages API error {response.status_code}: {error_text[:500]}"
                )
                raise httpx.HTTPStatusError(
                    f"Copilot Anthropic Messages API error: {response.status_code}",
                    request=response.request,
                    response=response,
                )

            # Capture response headers for quota tracking
            captured_headers = {k.lower(): v for k, v in response.headers.items()}
            is_first_chunk = True

            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("event: "):
                    current_event = line[len("event: ") :].strip()
                    continue
                if not line.startswith("data: "):
                    continue

                raw = line[len("data: ") :].strip()
                if not raw or raw == "[DONE]":
                    continue

                try:
                    evt = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                # Fallback: if Copilot returns OpenAI-style streaming even on /v1/messages.
                if isinstance(evt, dict) and "choices" in evt:
                    for choice in evt.get("choices", []):
                        delta = choice.get("delta", {}) or {}
                        if "content" in delta:
                            chunk = litellm.ModelResponse(
                                id=response_id,
                                created=created,
                                model=f"github_copilot/{model}",
                                object="chat.completion.chunk",
                                choices=[
                                    {
                                        "index": choice.get("index", 0),
                                        "delta": {
                                            "content": delta.get("content"),
                                            "role": delta.get("role", "assistant"),
                                        },
                                        "finish_reason": None,
                                    }
                                ],
                            )
                            if is_first_chunk:
                                chunk._response_headers = captured_headers
                                is_first_chunk = False
                            yield chunk
                        if "reasoning_content" in delta:
                            chunk = litellm.ModelResponse(
                                id=response_id,
                                created=created,
                                model=f"github_copilot/{model}",
                                object="chat.completion.chunk",
                                choices=[
                                    {
                                        "index": choice.get("index", 0),
                                        "delta": {
                                            "reasoning_content": delta.get(
                                                "reasoning_content"
                                            ),
                                            "role": delta.get("role", "assistant"),
                                        },
                                        "finish_reason": None,
                                    }
                                ],
                            )
                            if is_first_chunk:
                                chunk._response_headers = captured_headers
                                is_first_chunk = False
                            yield chunk
                        if "tool_calls" in delta:
                            chunk = litellm.ModelResponse(
                                id=response_id,
                                created=created,
                                model=f"github_copilot/{model}",
                                object="chat.completion.chunk",
                                choices=[
                                    {
                                        "index": choice.get("index", 0),
                                        "delta": {
                                            "tool_calls": delta.get("tool_calls")
                                        },
                                        "finish_reason": None,
                                    }
                                ],
                            )
                            if is_first_chunk:
                                chunk._response_headers = captured_headers
                                is_first_chunk = False
                            yield chunk
                    continue

                event_type = evt.get("type") or current_event

                if event_type == "message_start":
                    msg = evt.get("message") or {}
                    if isinstance(msg, dict) and msg.get("id"):
                        response_id = msg["id"]
                    usage = msg.get("usage") or {}
                    if isinstance(usage, dict):
                        input_tokens = (
                            usage.get("input_tokens", input_tokens) or input_tokens
                        )
                        output_tokens = (
                            usage.get("output_tokens", output_tokens) or output_tokens
                        )
                    continue

                if event_type == "content_block_start":
                    idx = evt.get("index")
                    block = evt.get("content_block") or {}
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_use":
                        tc_index = len(tool_calls)
                        call_id = block.get("id") or f"toolu_{uuid.uuid4().hex[:12]}"
                        name = block.get("name") or ""
                        tool_calls[tc_index] = {
                            "index": tc_index,
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": ""},
                        }
                        if isinstance(idx, int):
                            tool_block_index_to_tc_index[idx] = tc_index
                        chunk = litellm.ModelResponse(
                            id=response_id,
                            created=created,
                            model=f"github_copilot/{model}",
                            object="chat.completion.chunk",
                            choices=[
                                {
                                    "index": 0,
                                    "delta": {"tool_calls": [tool_calls[tc_index]]},
                                    "finish_reason": None,
                                }
                            ],
                        )
                        if is_first_chunk:
                            chunk._response_headers = captured_headers
                            is_first_chunk = False
                        yield chunk
                    continue

                if event_type == "content_block_delta":
                    idx = evt.get("index")
                    delta = evt.get("delta") or {}
                    if not isinstance(delta, dict):
                        continue
                    dtype = delta.get("type")

                    if dtype == "thinking_delta":
                        thinking = delta.get("thinking")
                        if thinking:
                            chunk = litellm.ModelResponse(
                                id=response_id,
                                created=created,
                                model=f"github_copilot/{model}",
                                object="chat.completion.chunk",
                                choices=[
                                    {
                                        "index": 0,
                                        "delta": {
                                            "reasoning_content": thinking,
                                            "role": "assistant",
                                        },
                                        "finish_reason": None,
                                    }
                                ],
                            )
                            if is_first_chunk:
                                chunk._response_headers = captured_headers
                                is_first_chunk = False
                            yield chunk
                        continue

                    if dtype == "text_delta":
                        text = delta.get("text")
                        if text:
                            chunk = litellm.ModelResponse(
                                id=response_id,
                                created=created,
                                model=f"github_copilot/{model}",
                                object="chat.completion.chunk",
                                choices=[
                                    {
                                        "index": 0,
                                        "delta": {"content": text, "role": "assistant"},
                                        "finish_reason": None,
                                    }
                                ],
                            )
                            if is_first_chunk:
                                chunk._response_headers = captured_headers
                                is_first_chunk = False
                            yield chunk
                        continue

                    if dtype == "input_json_delta":
                        partial = delta.get("partial_json")
                        if partial and isinstance(idx, int):
                            tc_index = tool_block_index_to_tc_index.get(idx)
                            if tc_index is not None and tc_index in tool_calls:
                                tool_calls[tc_index]["function"]["arguments"] += partial
                                chunk = litellm.ModelResponse(
                                    id=response_id,
                                    created=created,
                                    model=f"github_copilot/{model}",
                                    object="chat.completion.chunk",
                                    choices=[
                                        {
                                            "index": 0,
                                            "delta": {
                                                "tool_calls": [
                                                    {
                                                        "index": tc_index,
                                                        "id": tool_calls[tc_index][
                                                            "id"
                                                        ],
                                                        "type": "function",
                                                        "function": {
                                                            "name": tool_calls[
                                                                tc_index
                                                            ]["function"]["name"],
                                                            "arguments": partial,
                                                        },
                                                    }
                                                ]
                                            },
                                            "finish_reason": None,
                                        }
                                    ],
                                )
                                if is_first_chunk:
                                    chunk._response_headers = captured_headers
                                    is_first_chunk = False
                                yield chunk
                        continue

                if event_type == "message_delta":
                    delta = evt.get("delta") or {}
                    usage = evt.get("usage") or {}
                    stop_reason = None
                    if isinstance(delta, dict):
                        stop_reason = delta.get("stop_reason")
                    if isinstance(usage, dict):
                        input_tokens = (
                            usage.get("input_tokens", input_tokens) or input_tokens
                        )
                        output_tokens = (
                            usage.get("output_tokens", output_tokens) or output_tokens
                        )

                    finish_reason = self._map_anthropic_stop_reason_to_finish_reason(
                        stop_reason
                    )
                    chunk = litellm.ModelResponse(
                        id=response_id,
                        created=created,
                        model=f"github_copilot/{model}",
                        object="chat.completion.chunk",
                        choices=[
                            {"index": 0, "delta": {}, "finish_reason": finish_reason}
                        ],
                        usage={
                            "prompt_tokens": input_tokens,
                            "completion_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                        },
                    )
                    if is_first_chunk:
                        chunk._response_headers = captured_headers
                        is_first_chunk = False
                    yield chunk
                    continue

                if event_type == "message_stop":
                    break

                if event_type == "error":
                    err = evt.get("error") if isinstance(evt, dict) else None
                    raise RuntimeError(f"Copilot Anthropic stream error: {err or evt}")

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
            # Handle reasoning fields (OpenCode parity)
            if "reasoning_content" in message:
                choices[i]["message"]["reasoning_content"] = message[
                    "reasoning_content"
                ]
            if "reasoning_text" in message:
                choices[i]["message"]["reasoning_text"] = message["reasoning_text"]
                if "reasoning_content" not in message:
                    choices[i]["message"]["reasoning_content"] = message[
                        "reasoning_text"
                    ]
            if "reasoning_opaque" in message:
                choices[i]["message"]["reasoning_opaque"] = message["reasoning_opaque"]

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

        # Attach response headers for quota tracking
        response_obj._response_headers = {
            k.lower(): v for k, v in response.headers.items()
        }

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

        # Disable compression for streaming to get smooth token-by-token output.
        # Without this, gzip compression causes buffering until complete blocks
        # can be decompressed, resulting in chunky streaming.
        stream_headers = {**headers, "Accept-Encoding": "identity"}

        async with client.stream(
            "POST",
            endpoint,
            headers=stream_headers,
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

            # Capture response headers for quota tracking
            captured_headers = {k.lower(): v for k, v in response.headers.items()}
            is_first_chunk = True

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
                        # Attach headers to first chunk for quota tracking
                        if is_first_chunk:
                            chunk._response_headers = captured_headers
                            is_first_chunk = False
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
                        # Attach headers to first chunk for quota tracking
                        if is_first_chunk:
                            chunk._response_headers = captured_headers
                            is_first_chunk = False
                        yield chunk

                    # Handle reasoning_text delta (OpenCode parity)
                    if "reasoning_text" in delta:
                        chunk = litellm.ModelResponse(
                            id=response_id,
                            created=created,
                            model=f"github_copilot/{model}",
                            object="chat.completion.chunk",
                            choices=[
                                {
                                    "index": index,
                                    "delta": {
                                        "reasoning_content": delta["reasoning_text"],
                                        "reasoning_text": delta["reasoning_text"],
                                        "role": delta.get("role", "assistant"),
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        )
                        if is_first_chunk:
                            chunk._response_headers = captured_headers
                            is_first_chunk = False
                        yield chunk

                    # Handle reasoning_opaque delta (OpenCode parity - multi-turn continuity)
                    if "reasoning_opaque" in delta:
                        chunk = litellm.ModelResponse(
                            id=response_id,
                            created=created,
                            model=f"github_copilot/{model}",
                            object="chat.completion.chunk",
                            choices=[
                                {
                                    "index": index,
                                    "delta": {
                                        "reasoning_opaque": delta["reasoning_opaque"]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        )
                        if is_first_chunk:
                            chunk._response_headers = captured_headers
                            is_first_chunk = False
                        yield chunk

                    # Handle thinking delta (Anthropic's native format - may come from Copilot for Claude)
                    if "thinking" in delta:
                        thinking_content = delta["thinking"]
                        if thinking_content:
                            chunk = litellm.ModelResponse(
                                id=response_id,
                                created=created,
                                model=f"github_copilot/{model}",
                                object="chat.completion.chunk",
                                choices=[
                                    {
                                        "index": index,
                                        "delta": {
                                            "reasoning_content": thinking_content,
                                            "role": delta.get("role", "assistant"),
                                        },
                                        "finish_reason": None,
                                    }
                                ],
                            )
                            if is_first_chunk:
                                chunk._response_headers = captured_headers
                                is_first_chunk = False
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
                            # Include index in each tool call for proper accumulation
                            tool_calls_list = [
                                {"index": i, **current_tool_calls[i]}
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
                            # Attach headers to first chunk for quota tracking
                            if is_first_chunk:
                                chunk._response_headers = captured_headers
                                is_first_chunk = False
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
                        # Attach headers to first chunk for quota tracking
                        if is_first_chunk:
                            final_chunk._response_headers = captured_headers
                            is_first_chunk = False
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
        call_id_by_name: Dict[str, List[str]] = {}

        for msg in messages or []:
            if not isinstance(msg, dict):
                continue
            role_raw = msg.get("role", "user")
            role = str(role_raw).strip().lower() if role_raw is not None else "user"
            content = msg.get("content")

            # Map system role to developer (Responses API convention)
            if role == "system":
                role = "developer"

            # Handle assistant messages (previous responses)
            if role == "assistant":
                # Get reasoning fields for multi-turn support (OpenCode parity)
                reasoning_text = msg.get("reasoning_text") or msg.get(
                    "reasoning_content"
                )
                reasoning_opaque = msg.get("reasoning_opaque")

                # Convert to Responses API message format
                if isinstance(content, str):
                    message_item: Dict[str, Any] = {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content}],
                    }
                    # Add reasoning fields if present (OpenCode parity)
                    if reasoning_text:
                        message_item["reasoning_text"] = reasoning_text
                    if reasoning_opaque:
                        message_item["reasoning_opaque"] = reasoning_opaque
                    input_items.append(message_item)
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
                    message_item = {
                        "type": "message",
                        "role": "assistant",
                        "content": output_content,
                    }
                    # Add reasoning fields if present (OpenCode parity)
                    if reasoning_text:
                        message_item["reasoning_text"] = reasoning_text
                    if reasoning_opaque:
                        message_item["reasoning_opaque"] = reasoning_opaque
                    input_items.append(message_item)
                tool_calls = msg.get("tool_calls") or []
                for tool_call in tool_calls:
                    if tool_call.get("type") != "function":
                        continue
                    function = tool_call.get("function", {})
                    call_id = tool_call.get("id", "")
                    func_name = (
                        function.get("name", "") if isinstance(function, dict) else ""
                    )
                    # Skip invalid tool calls missing required fields
                    if not call_id or not func_name:
                        lib_logger.warning(
                            f"[GitHubCopilot] Skipping invalid tool call in Responses API conversion: "
                            f"id={call_id!r}, name={func_name!r}"
                        )
                        continue
                    call_id_by_name.setdefault(func_name, []).append(call_id)
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": func_name,
                            "arguments": function.get("arguments", ""),
                        }
                    )
                continue

            # Handle tool/function messages (function outputs)
            if role in ("tool", "function"):
                tool_call_id = (
                    msg.get("tool_call_id") or msg.get("call_id") or msg.get("id") or ""
                )
                tool_name = msg.get("name") or msg.get("tool_name") or ""
                if not tool_call_id and tool_name:
                    queued = call_id_by_name.get(tool_name) or []
                    if queued:
                        tool_call_id = queued.pop(0)
                        if not queued:
                            call_id_by_name.pop(tool_name, None)
                if isinstance(content, str):
                    output = content
                elif content is None:
                    output = ""
                else:
                    output = json.dumps(content)
                if tool_call_id:
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": tool_call_id,
                            "output": output,
                        }
                    )
                else:
                    lib_logger.warning(
                        "[GitHubCopilot] Tool output missing call_id; falling back to user message. name=%r",
                        tool_name,
                    )
                    input_items.append(
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": output}],
                        }
                    )
                continue

            if role not in ("user", "developer"):
                role = "user"

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
        }
        if stream:
            payload["stream"] = True

        # Copy over optional parameters with Responses API naming
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            payload["max_output_tokens"] = kwargs["max_tokens"]
        if "top_p" in kwargs and kwargs["top_p"] is not None:
            payload["top_p"] = kwargs["top_p"]
        # OpenCode parity parameters
        if "verbosity" in kwargs and kwargs["verbosity"] is not None:
            payload["verbosity"] = kwargs["verbosity"]

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

        # Handle reasoning parameters for GPT-5/o-series (Responses API uses reasoning.effort)
        reasoning_effort = kwargs.get("reasoning_effort")
        reasoning_summary = kwargs.get("reasoning_summary") or kwargs.get(
            "reasoningSummary"
        )

        if reasoning_effort is not None:
            thinking_config = _map_reasoning_effort_to_config(reasoning_effort, model)
            lib_logger.debug(
                "[GitHubCopilot] Mapped thinking_config: %s",
                json.dumps(thinking_config, default=str),
            )
            payload.update(thinking_config)

        # Add reasoningSummary for GPT models (OpenCode parity)
        if reasoning_summary is not None:
            if "reasoning" in payload:
                payload["reasoning"]["summary"] = reasoning_summary
            else:
                payload["reasoning"] = {"summary": reasoning_summary}

        # Structured output
        if "response_format" in kwargs and kwargs["response_format"] is not None:
            payload["text"] = {"format": kwargs["response_format"]}

        endpoint = f"{api_base}/responses"

        lib_logger.debug(
            "[GitHubCopilot] Responses API request: model=%s, reasoning=%s",
            model,
            json.dumps(payload.get("reasoning"), default=str),
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

        # Attach response headers for quota tracking
        response_obj._response_headers = {
            k.lower(): v for k, v in response.headers.items()
        }

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

        # Track tool call indices - Responses API may not include index in events
        tool_call_index_counter = 0

        # Track if any tool calls were yielded (for correct finish_reason)
        tool_calls_yielded = False

        # Disable compression for streaming to get smooth token-by-token output.
        # Without this, gzip compression causes buffering until complete blocks
        # can be decompressed, resulting in chunky streaming.
        stream_headers = {**headers, "Accept-Encoding": "identity"}

        async with client.stream(
            "POST",
            endpoint,
            headers=stream_headers,
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

            # Capture response headers for quota tracking
            captured_headers = {k.lower(): v for k, v in response.headers.items()}
            is_first_chunk = True

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

                # Handle different event types
                event_type = evt.get("type", "")

                # Extract response ID from response.created event only
                # Other events like response.output_item.done have item IDs, not response IDs
                if event_type == "response.created":
                    if evt.get("response", {}).get("id"):
                        response_id = evt["response"]["id"]
                    elif evt.get("id"):
                        response_id = evt["id"]

                if event_type == "response.output_item.done":
                    item = evt.get("item", {})
                    if item.get("type") == "reasoning":
                        lib_logger.debug(
                            "[GitHubCopilot] reasoning item: %s",
                            json.dumps(item, default=str)[:300],
                        )

                if event_type == "response.content_part.delta":
                    content_part = evt.get("part", {})
                    if content_part.get("type") == "reasoning":
                        lib_logger.debug(
                            "[GitHubCopilot] reasoning part delta: %s",
                            str(content_part.get("text", ""))[:200],
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
                        if is_first_chunk:
                            chunk._response_headers = captured_headers
                            is_first_chunk = False
                        yield chunk

                # Handle reasoning summary delta events (GPT-5.2 and o-series)
                elif event_type == "response.reasoning_summary_text.delta":
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
                                        "reasoning_content": delta_text,
                                        "role": "assistant",
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        )
                        if is_first_chunk:
                            chunk._response_headers = captured_headers
                            is_first_chunk = False
                        yield chunk

                # Handle response completion events
                elif event_type in ("response.done", "response.completed"):
                    # Determine finish_reason based on what was emitted
                    # If tool calls were yielded, clients expect "tool_calls" not "stop"
                    final_finish_reason = "tool_calls" if tool_calls_yielded else "stop"
                    final_chunk = litellm.ModelResponse(
                        id=response_id,
                        created=created,
                        model=f"github_copilot/{model}",
                        object="chat.completion.chunk",
                        choices=[
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": final_finish_reason,
                            }
                        ],
                    )
                    if is_first_chunk:
                        final_chunk._response_headers = captured_headers
                        is_first_chunk = False
                    yield final_chunk
                    break  # Exit stream loop - response is complete

                elif event_type == "response.output_item.done":
                    item = evt.get("item", {})
                    item_type = item.get("type")
                    if item_type in ("function_call", "tool_call"):
                        call_id = item.get("call_id") or item.get("id")
                        # Use our own counter to ensure unique indices for each tool call
                        # The API may not include index or may send all as index 0
                        call_index = tool_call_index_counter
                        tool_call_index_counter += 1
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
                            if is_first_chunk:
                                chunk._response_headers = captured_headers
                                is_first_chunk = False
                            tool_calls_yielded = (
                                True  # Mark that tool calls were emitted
                            )
                            yield chunk
                    elif item_type == "reasoning":
                        # Skip - reasoning was already streamed via response.reasoning_summary_text.delta
                        # Emitting it again here would cause duplicates
                        pass
                    elif item_type == "message":
                        # Skip - message content was already streamed via response.output_text.delta
                        # Emitting it again here would cause duplicates (same as reasoning above)
                        pass
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
                            if is_first_chunk:
                                chunk._response_headers = captured_headers
                                is_first_chunk = False
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
                            if is_first_chunk:
                                chunk._response_headers = captured_headers
                                is_first_chunk = False
                            yield chunk

    # =========================================================================
    # QUOTA TRACKING (PROACTIVE FETCH)
    # =========================================================================

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        """
        Configure periodic quota refresh for GitHub Copilot.

        Returns:
            Background job configuration for quota refresh
        """
        return {
            "interval": COPILOT_QUOTA_REFRESH_INTERVAL,
            "name": "github_copilot_quota_refresh",
            "run_on_start": True,
        }

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
        """
        Refresh quota usage for all credentials in parallel.

        Fetches quota from GitHub's internal Copilot API:
        https://api.github.com/copilot_internal/user

        Args:
            usage_manager: UsageManager instance
            credentials: List of credential paths
        """
        semaphore = asyncio.Semaphore(QUOTA_FETCH_CONCURRENCY)

        async def refresh_single_credential(
            credential_path: str, client: httpx.AsyncClient
        ) -> None:
            async with semaphore:
                try:
                    quota_data = await self.fetch_copilot_quota(credential_path, client)

                    if quota_data.get("status") == "success":
                        # Extract premium_interactions quota (the main quota for premium requests)
                        premium = quota_data.get("premium_interactions", {})
                        entitlement = premium.get("entitlement", 0)
                        remaining = premium.get("remaining", 0)
                        reset_date = quota_data.get("reset_date")

                        # Parse reset date to Unix timestamp
                        reset_ts = None
                        if reset_date:
                            try:
                                from datetime import datetime

                                # Parse ISO format date (YYYY-MM-DD)
                                dt = datetime.strptime(reset_date, "%Y-%m-%d")
                                reset_ts = dt.timestamp()
                            except (ValueError, TypeError):
                                lib_logger.debug(
                                    f"Could not parse reset date: {reset_date}"
                                )

                        # Calculate used count
                        quota_used = entitlement - remaining if entitlement > 0 else 0

                        # Store baseline in usage manager using the "requests" group
                        # This is a virtual model for credential-level tracking
                        await usage_manager.update_quota_baseline(
                            credential_path,
                            "github_copilot/_quota",  # Virtual model for credential-level tracking
                            quota_max_requests=entitlement,
                            quota_reset_ts=reset_ts,
                            quota_used=quota_used,
                            quota_group="requests",  # Use "requests" as the quota group
                        )

                        lib_logger.debug(
                            f"Updated GitHub Copilot quota baseline: "
                            f"{remaining}/{entitlement} remaining, "
                            f"resets {reset_date}"
                        )

                except Exception as e:
                    lib_logger.warning(f"Failed to refresh GitHub Copilot quota: {e}")

        # Fetch all credentials in parallel with shared HTTP client
        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [
                refresh_single_credential(cred_path, client)
                for cred_path in credentials
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def fetch_copilot_quota(
        self,
        credential_path: str,
        client: httpx.AsyncClient,
    ) -> Dict[str, Any]:
        """
        Fetch quota information from GitHub's internal Copilot API.

        Calls: GET https://api.github.com/copilot_internal/user

        Response format:
        {
            "copilot_plan": "pro",
            "quota_reset_date": "2026-02-01",
            "quota_snapshots": {
                "premium_interactions": {
                    "entitlement": 300,
                    "percent_remaining": 85.0,
                    "remaining": 255,
                    "unlimited": false
                },
                ...
            }
        }

        Args:
            credential_path: Path to credential file or env:// path
            client: HTTP client instance

        Returns:
            Dict with quota info or error status
        """
        try:
            # Get OAuth token for GitHub API
            auth_header = await self.get_auth_header(credential_path)

            # Build headers for GitHub API (use consistent User-Agent)
            headers = {
                **auth_header,
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            }

            # Fetch quota from GitHub internal API
            response = await client.get(
                f"{GITHUB_API_BASE}/copilot_internal/user",
                headers=headers,
            )

            if response.status_code == 401:
                lib_logger.warning(
                    f"GitHub Copilot quota fetch unauthorized - token may be invalid"
                )
                return {"status": "error", "error": "unauthorized"}

            if response.status_code != 200:
                lib_logger.warning(
                    f"GitHub Copilot quota fetch failed: HTTP {response.status_code}"
                )
                return {"status": "error", "error": f"HTTP {response.status_code}"}

            data = response.json()

            # Extract quota information
            quota_snapshots = data.get("quota_snapshots", {})
            premium = quota_snapshots.get("premium_interactions", {})

            result = {
                "status": "success",
                "copilot_plan": data.get("copilot_plan"),
                "reset_date": data.get("quota_reset_date"),
                "premium_interactions": {
                    "entitlement": premium.get("entitlement", 0),
                    "remaining": premium.get("remaining", 0),
                    "percent_remaining": premium.get("percent_remaining", 0),
                    "unlimited": premium.get("unlimited", False),
                },
            }

            # Also extract chat and completions if available
            if "chat" in quota_snapshots:
                chat = quota_snapshots["chat"]
                result["chat"] = {
                    "entitlement": chat.get("entitlement", 0),
                    "remaining": chat.get("remaining", 0),
                    "unlimited": chat.get("unlimited", False),
                }

            if "completions" in quota_snapshots:
                completions = quota_snapshots["completions"]
                result["completions"] = {
                    "entitlement": completions.get("entitlement", 0),
                    "remaining": completions.get("remaining", 0),
                    "unlimited": completions.get("unlimited", False),
                }

            return result

        except httpx.TimeoutException:
            lib_logger.warning("GitHub Copilot quota fetch timed out")
            return {"status": "error", "error": "timeout"}
        except Exception as e:
            lib_logger.warning(f"GitHub Copilot quota fetch error: {e}")
            return {"status": "error", "error": str(e)}
