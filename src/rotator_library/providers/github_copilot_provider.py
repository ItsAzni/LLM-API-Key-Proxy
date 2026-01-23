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

import logging
from typing import List

import httpx

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
