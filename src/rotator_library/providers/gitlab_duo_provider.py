# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
GitLab Duo Provider

Provider implementation for GitLab Duo AI Gateway, enabling access to
Anthropic Claude and OpenAI GPT models through GitLab's AI infrastructure
using GitLab Personal Access Tokens (PATs) or OAuth 2.0 (PKCE).

Key Features:
- Two-step authentication: PAT/OAuth -> short-lived direct access token
- OAuth 2.0 with PKCE (Authorization Code flow, no client_secret needed)
- Claude models via Anthropic Messages API proxy
- GPT models via OpenAI Chat Completions API proxy
- Extended thinking support for Claude models
- Token caching with automatic refresh

Authentication methods:
- PAT: PRIVATE-TOKEN header with Personal Access Token
- OAuth: Authorization: Bearer header with OAuth access token (auto-refreshed)

API Flow:
1. POST {instanceUrl}/api/v4/ai/third_party_agents/direct_access -> short-lived token
2. POST {aiGatewayUrl}/ai/v1/proxy/anthropic/v1/messages (Claude)
   POST {aiGatewayUrl}/ai/v1/proxy/openai/v1/chat/completions (GPT)

Based on reverse-engineering of @gitlab/gitlab-ai-provider and
@gitlab/opencode-gitlab-auth npm packages.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import secrets
import time
import uuid
import webbrowser
from contextlib import suppress
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from urllib.parse import urlencode, urlparse, parse_qs

import httpx
import litellm

from ..timeout_config import TimeoutConfig
from ..utils.resilient_io import safe_write_json
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default GitLab instance URL
DEFAULT_INSTANCE_URL = "https://gitlab.com"

# Default AI Gateway URL (the SDK appends /ai/v1/proxy/... paths to this)
DEFAULT_AI_GATEWAY_URL = "https://cloud.gitlab.com"

# Token cache duration (25 minutes; tokens expire at 30 min)
TOKEN_CACHE_DURATION = 25 * 60

# =============================================================================
# OAUTH CONFIGURATION
# =============================================================================

# Default OAuth client ID from GitLab VS Code extension (public client).
# This app has http://127.0.0.1:8080/callback registered as a redirect URI.
# Override via GITLAB_OAUTH_CLIENT_ID for self-managed instances.
DEFAULT_OAUTH_CLIENT_ID = (
    "1d89f9fdb23ee96d4e603201f6861dab6e143c5c3c00469a018a2d94bdc03d4e"
)

# OAuth scopes
OAUTH_SCOPES = ["api"]

# OAuth callback configuration — port 8080 matches the registered redirect URI
# IMPORTANT: Must use 127.0.0.1 (NOT localhost) to match the registered redirect URI
DEFAULT_OAUTH_CALLBACK_PORT = 8080
OAUTH_CALLBACK_PATH = "/callback"

# Token refresh buffer (refresh 5 minutes before expiry)
OAUTH_REFRESH_BUFFER = 5 * 60

# =============================================================================
# CREDIT COST TABLE
# =============================================================================

# Credits consumed per request for each model
# Source: https://docs.gitlab.com/subscriptions/gitlab_credits/
CREDIT_COSTS: Dict[str, float] = {
    # Anthropic Claude models
    "claude-opus-4-6": 1 / 0.7,  # 0.7 requests/credit → ~1.43 credits/req
    "claude-opus-4-5": 1 / 1.2,  # 1.2 requests/credit → ~0.83 credits/req
    "claude-sonnet-4-6": 1 / 1.1,  # ~0.91 credits/req
    "claude-sonnet-4-5": 1 / 2.0,  # 2.0 requests/credit → 0.5 credits/req
    "claude-haiku-4-5": 1 / 6.7,  # 6.7 requests/credit → ~0.15 credits/req
    # OpenAI GPT models
    "gpt-5-2": 1 / 2.5,  # 2.5 requests/credit → 0.4 credits/req
    "gpt-5-1": 1 / 3.3,  # 3.3 requests/credit → ~0.30 credits/req
    "gpt-5-mini": 1 / 8.0,  # 8.0 requests/credit → 0.125 credits/req
    "gpt-5-2-codex": 1 / 3.3,  # codex variant → ~0.30 credits/req
    "gpt-5-codex": 1 / 3.3,  # codex variant → ~0.30 credits/req
}

# Default credits per account (Ultimate = 24/user/month, Premium = 12)
DEFAULT_CREDITS_PER_ACCOUNT = 24.0

# Maximum exhaustion strikes before auto-removing a credential
MAX_EXHAUSTION_STRIKES = 3

# Cooldown applied when a real credit exhaustion 402 is detected (seconds)
CREDIT_EXHAUSTION_COOLDOWN = 300

# Anthropic API version
ANTHROPIC_VERSION = "2023-06-01"

# Anthropic beta for interleaved thinking
ANTHROPIC_BETA = "interleaved-thinking-2025-05-14"

# Interleaved thinking reminder — injected into last user message during tool loops
# to ensure Claude emits thinking blocks on every response
INTERLEAVED_THINKING_REMINDER = """<system-reminder>
# Interleaved Thinking - Active

You MUST emit a thinking block on EVERY response:
- **Before** any action (reason about what to do)
- **After** any result (analyze before next step)

Never skip thinking, even on follow-up responses. Ultrathink
</system-reminder>"""

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Model mappings: proxy_name -> (backend_model_name, provider_type)
# The backend_model_name is the actual model ID sent in the API request body.
# Source: @gitlab/gitlab-ai-provider MODEL_MAPPINGS + https://models.dev
MODEL_MAP: Dict[str, Tuple[str, str]] = {
    # Anthropic Claude models
    "claude-opus-4-6": ("claude-opus-4-6", "anthropic"),
    "claude-opus-4-5": ("claude-opus-4-5-20251101", "anthropic"),
    "claude-sonnet-4-6": ("claude-sonnet-4-6", "anthropic"),
    "claude-sonnet-4-5": ("claude-sonnet-4-5-20250929", "anthropic"),
    "claude-haiku-4-5": ("claude-haiku-4-5-20251001", "anthropic"),
    # OpenAI GPT models (chat completions)
    "gpt-5-2": ("gpt-5.2-2025-12-11", "openai"),
    "gpt-5-1": ("gpt-5.1-2025-11-13", "openai"),
    "gpt-5-mini": ("gpt-5-mini-2025-08-07", "openai"),
    # OpenAI GPT models (codex/responses API - routed as chat completions)
    "gpt-5-2-codex": ("gpt-5.2-codex", "openai"),
    "gpt-5-codex": ("gpt-5-codex", "openai"),
}

# Thinking budget mapping for Claude reasoning effort (legacy models: 4.5, haiku)
THINKING_BUDGET_MAP = {
    "auto": 31999,
    "minimal": 1024,
    "low": 4096,
    "low_medium": 8192,
    "medium": 16000,
    "medium_high": 24000,
    "high": 31999,
    "xhigh": 31999,
    "max": 31999,
}

# 4.6 models use adaptive thinking (type: "adaptive" + output_config.effort)
ADAPTIVE_THINKING_MODELS = frozenset({"claude-opus-4-6", "claude-sonnet-4-6"})

# Maps internal granular reasoning_effort names to Anthropic's effort levels
REASONING_TO_EFFORT_MAP = {
    "minimal": "low",
    "low": "low",
    "low_medium": "medium",
    "medium": "medium",
    "medium_high": "high",
    "high": "high",
    "xhigh": "max",
    "max": "max",
    "auto": "high",
}

# Converts legacy budget_tokens to effort levels for 4.6 models
BUDGET_TO_EFFORT_THRESHOLDS = [
    (4096, "low"),
    (12000, "medium"),
    (24000, "high"),
    (999999, "max"),
]


def _get_instance_url() -> str:
    return os.getenv("GITLAB_DUO_INSTANCE_URL", DEFAULT_INSTANCE_URL).rstrip("/")


def _get_ai_gateway_url() -> str:
    return os.getenv("GITLAB_AI_GATEWAY_URL", DEFAULT_AI_GATEWAY_URL).rstrip("/")


# =============================================================================
# PROVIDER IMPLEMENTATION
# =============================================================================


class GitLabDuoProvider(ProviderInterface):
    """
    Provider implementation for GitLab Duo AI Gateway.

    Uses two-step authentication:
    1. Exchange PAT for short-lived direct access token
    2. Call AI Gateway proxy endpoints with the token
    """

    # =========================================================================
    # PROVIDER CONFIGURATION
    # =========================================================================

    provider_env_name: str = "gitlab_duo"
    skip_cost_calculation: bool = True
    default_rotation_mode: str = "sequential"

    tier_priorities = {
        "duo-enterprise": 1,
        "duo-pro": 1,
        "duo-trial": 2,
    }
    default_tier_priority: int = 2

    model_quota_groups = {
        "claude": [
            "claude-opus-4-6",
            "claude-opus-4-5",
            "claude-sonnet-4-6",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
        ],
        "gpt": [
            "gpt-5-2",
            "gpt-5-2-codex",
            "gpt-5-1",
            "gpt-5-codex",
            "gpt-5-mini",
        ],
    }

    # Per-credential token cache: api_key -> (token_data, expires_at)
    _token_cache: Dict[str, Tuple[dict, float]] = {}

    # In-memory OAuth credential cache: cred_path -> loaded creds dict
    _oauth_cred_cache: Dict[str, dict] = {}

    # Credit tracking state
    # Per-credential accumulated credit usage this month
    _credit_usage: Dict[str, float] = {}
    # Per-credential credit limit (from env or default)
    _credit_limits: Dict[str, float] = {}
    # Per-credential 402 "insufficient_credits" strike counter
    _exhaustion_strikes: Dict[str, int] = {}
    # Credentials that have been auto-removed due to exhaustion
    _removed_credentials: set = set()
    # Optional async callback fired when a credential is exhausted and removed
    _credential_exhausted_callback: Optional[Callable[..., Awaitable[None]]] = None

    # =========================================================================
    # PROVIDER INTERFACE
    # =========================================================================

    def has_custom_logic(self) -> bool:
        return True

    def should_remove_credential(self, credential: str) -> bool:
        """Check if a credential has been flagged for removal due to exhaustion."""
        return credential in self._removed_credentials

    def set_credential_exhausted_callback(
        self, callback: Callable[..., Awaitable[None]]
    ) -> None:
        """Register an async callback invoked when a credential is exhausted and removed.

        The callback receives a single argument: the credential string (file path or PAT).
        It is fired via ``asyncio.create_task`` so it does not block the request path.
        """
        self._credential_exhausted_callback = callback

    async def initialize_token(
        self,
        creds_or_path: Union[str, dict],
        force_interactive: bool = False,
    ) -> dict:
        """
        Initialize and validate an OAuth credential on startup.

        For PAT credentials (non-file strings), this is a no-op.
        For OAuth credentials (file paths), loads and refreshes if expired.
        """
        if isinstance(creds_or_path, str) and self._is_oauth_credential(creds_or_path):
            creds = self._load_oauth_credentials(creds_or_path)
            expires_at = creds.get("expires_at", 0)
            if time.time() >= (expires_at - OAUTH_REFRESH_BUFFER):
                lib_logger.info(
                    "[GitLabDuo] OAuth token expired on startup, refreshing..."
                )
                creds = await self._refresh_oauth_token(creds_or_path, creds)
            return creds
        # PAT credentials don't need initialization
        return {"access_token": creds_or_path if isinstance(creds_or_path, str) else ""}

    async def get_user_info(self, creds_or_path: Union[str, dict]) -> dict:
        """
        Get user info for an OAuth credential.

        Returns dict with 'email' key for deduplication.
        PAT credentials return empty dict (no user info available).
        """
        if isinstance(creds_or_path, str) and self._is_oauth_credential(creds_or_path):
            creds = self._load_oauth_credentials(creds_or_path)
            instance_url = creds.get("instance_url", _get_instance_url())
            access_token = creds.get("access_token")
            if access_token:
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            f"{instance_url}/api/v4/user",
                            headers={"Authorization": f"Bearer {access_token}"},
                            timeout=10.0,
                        )
                        if response.status_code == 200:
                            data = response.json()
                            return {
                                "email": data.get("email")
                                or data.get("username", "unknown")
                            }
                except Exception as e:
                    lib_logger.debug("[GitLabDuo] Failed to fetch user info: %s", e)
        return {}

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        return [f"gitlab_duo/{name}" for name in MODEL_MAP]

    # =========================================================================
    # TOKEN MANAGEMENT
    # =========================================================================

    async def _get_or_refresh_token(
        self, api_key: str, client: httpx.AsyncClient
    ) -> dict:
        """Get cached token or fetch a new one."""
        cached = self._token_cache.get(api_key)
        if cached and cached[1] > time.time():
            return cached[0]

        token_data = await self._fetch_direct_access_token(api_key, client)
        self._token_cache[api_key] = (token_data, time.time() + TOKEN_CACHE_DURATION)
        return token_data

    def _invalidate_token(self, api_key: str) -> None:
        """Invalidate cached token (e.g., on 401)."""
        self._token_cache.pop(api_key, None)

    async def _fetch_direct_access_token(
        self, api_key: str, client: httpx.AsyncClient
    ) -> dict:
        """
        Exchange PAT or OAuth access token for a short-lived direct access token.

        POST {instanceUrl}/api/v4/ai/third_party_agents/direct_access

        Authentication:
        - PAT credentials: PRIVATE-TOKEN header
        - OAuth credentials: Authorization: Bearer header (token auto-refreshed)
        """
        instance_url = _get_instance_url()
        url = f"{instance_url}/api/v4/ai/third_party_agents/direct_access"

        # Determine auth method based on credential type
        if self._is_oauth_credential(api_key):
            access_token = await self._get_oauth_access_token(api_key)
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }
            lib_logger.debug("[GitLabDuo] Using OAuth token for direct access request")
        else:
            headers = {
                "PRIVATE-TOKEN": api_key,
                "Content-Type": "application/json",
            }

        body = {
            "feature_flags": {
                "duo_agent_platform_agentic_chat": True,
                "duo_agent_platform": True,
            }
        }

        lib_logger.debug("[GitLabDuo] Fetching direct access token from %s", url)

        response = await client.post(url, headers=headers, json=body, timeout=10.0)

        if response.status_code >= 400:
            error_text = response.text
            lib_logger.error(
                "[GitLabDuo] Direct access token fetch failed (%d): %s",
                response.status_code,
                error_text[:500],
            )
            raise httpx.HTTPStatusError(
                f"GitLab direct access token failed: {response.status_code}",
                request=response.request,
                response=response,
            )

        data = response.json()
        lib_logger.debug("[GitLabDuo] Direct access token obtained successfully")
        return data

    # =========================================================================
    # OAUTH CREDENTIAL MANAGEMENT
    # =========================================================================

    @staticmethod
    def _is_oauth_credential(credential: str) -> bool:
        """Check if a credential identifier refers to an OAuth JSON file."""
        if not credential:
            return False
        # OAuth credentials are stored as file paths to JSON files
        return credential.endswith(".json") or os.path.sep in credential

    async def _get_oauth_access_token(self, cred_path: str) -> str:
        """
        Get a valid OAuth access token from the credential file.

        Loads the credential JSON, checks expiry, and refreshes if needed.
        Returns the access_token string ready for use in API calls.
        """
        creds = self._load_oauth_credentials(cred_path)

        # Check if token needs refresh
        expires_at = creds.get("expires_at", 0)
        if time.time() >= (expires_at - OAUTH_REFRESH_BUFFER):
            lib_logger.info("[GitLabDuo] OAuth token expired, refreshing...")
            creds = await self._refresh_oauth_token(cred_path, creds)

        return creds["access_token"]

    def _load_oauth_credentials(self, cred_path: str) -> dict:
        """Load OAuth credentials from JSON file with caching."""
        # Check in-memory cache first
        cached = self._oauth_cred_cache.get(cred_path)
        if cached:
            return cached

        try:
            with open(cred_path, "r") as f:
                creds = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(
                f"[GitLabDuo] Failed to load OAuth credentials from {cred_path}: {e}"
            )

        required = ["access_token", "refresh_token"]
        for key in required:
            if key not in creds:
                raise ValueError(
                    f"[GitLabDuo] OAuth credential file missing '{key}': {cred_path}"
                )

        self._oauth_cred_cache[cred_path] = creds
        return creds

    async def _refresh_oauth_token(self, cred_path: str, creds: dict) -> dict:
        """
        Refresh an expired OAuth token using the refresh_token grant.

        POST {instanceUrl}/oauth/token
        """
        instance_url = creds.get("instance_url", _get_instance_url())
        client_id = creds.get(
            "client_id",
            os.getenv("GITLAB_OAUTH_CLIENT_ID", DEFAULT_OAUTH_CLIENT_ID),
        )
        if not client_id:
            raise ValueError(
                "[GitLabDuo] Cannot refresh OAuth token: no client_id in credential "
                "file and GITLAB_OAUTH_CLIENT_ID env var not set."
            )
        refresh_token = creds["refresh_token"]

        token_url = f"{instance_url}/oauth/token"
        payload = {
            "client_id": client_id,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                token_url,
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15.0,
            )

            if response.status_code >= 400:
                lib_logger.error(
                    "[GitLabDuo] OAuth token refresh failed (%d): %s",
                    response.status_code,
                    response.text[:500],
                )
                raise httpx.HTTPStatusError(
                    f"GitLab OAuth refresh failed: {response.status_code}",
                    request=response.request,
                    response=response,
                )

            data = response.json()

        # Update credentials
        creds["access_token"] = data["access_token"]
        creds["refresh_token"] = data.get("refresh_token", refresh_token)
        creds["expires_at"] = time.time() + data.get("expires_in", 7200)
        creds["token_type"] = data.get("token_type", "Bearer")

        # Persist updated credentials
        self._save_oauth_credentials(cred_path, creds)

        # Update cache
        self._oauth_cred_cache[cred_path] = creds

        lib_logger.info("[GitLabDuo] OAuth token refreshed successfully")
        return creds

    @staticmethod
    def _save_oauth_credentials(cred_path: str, creds: dict) -> None:
        """Persist OAuth credentials to JSON file."""
        try:
            safe_write_json(cred_path, creds, lib_logger)
            lib_logger.debug("[GitLabDuo] OAuth credentials saved to %s", cred_path)
        except Exception as e:
            lib_logger.warning("[GitLabDuo] Failed to save OAuth credentials: %s", e)

    # =========================================================================
    # OAUTH SETUP FLOW (Interactive, for credential_tool)
    # =========================================================================

    @staticmethod
    def _generate_pkce() -> Tuple[str, str]:
        """
        Generate PKCE code_verifier and code_challenge (S256).

        Returns: (code_verifier, code_challenge)
        """
        # 32 bytes = 43 base64url chars (RFC 7636 recommends 43-128)
        code_verifier = secrets.token_urlsafe(32)
        digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
        code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        return code_verifier, code_challenge

    @classmethod
    async def oauth_setup(
        cls,
        instance_url: Optional[str] = None,
        client_id: Optional[str] = None,
        callback_port: int = DEFAULT_OAUTH_CALLBACK_PORT,
        output_path: Optional[str] = None,
        auth_url_handler: Optional[Callable[[str], Awaitable[None]]] = None,
        auto_open_browser: bool = True,
    ) -> str:
        """
        Run the interactive OAuth 2.0 PKCE flow to obtain GitLab credentials.

        Opens the user's browser for authentication, starts a local callback
        server to receive the authorization code, and exchanges it for tokens.

        Args:
            instance_url: GitLab instance URL (default: from env or gitlab.com)
            client_id: OAuth client ID (default: from env or VS Code extension ID)
            callback_port: Local callback server port (default: 8080)
            output_path: Path to save the credential JSON file
            auth_url_handler: Optional async callback to handle auth URL opening
            auto_open_browser: Auto-open browser when no auth_url_handler is provided

        Returns:
            Path to the saved credential JSON file
        """
        if not instance_url:
            instance_url = _get_instance_url()
        if not client_id:
            client_id = os.getenv("GITLAB_OAUTH_CLIENT_ID", DEFAULT_OAUTH_CLIENT_ID)

        if not client_id:
            raise RuntimeError(
                "No OAuth client ID available. Set GITLAB_OAUTH_CLIENT_ID "
                "in your .env file. For self-managed GitLab instances, create "
                "an OAuth app at {instanceUrl}/-/profile/applications with "
                f"redirect URI: http://127.0.0.1:{callback_port}/callback"
            )

        instance_url = instance_url.rstrip("/")
        # IMPORTANT: Must use 127.0.0.1 (NOT localhost) to match the registered redirect URI
        redirect_uri = f"http://127.0.0.1:{callback_port}{OAUTH_CALLBACK_PATH}"

        # Generate PKCE
        code_verifier, code_challenge = cls._generate_pkce()

        # Generate CSRF state token
        state = secrets.token_urlsafe(32)

        # Build authorization URL
        auth_params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "state": state,
            "scope": " ".join(OAUTH_SCOPES),
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        auth_url = f"{instance_url}/oauth/authorize?{urlencode(auth_params)}"

        print(f"\n[GitLabDuo OAuth] Starting authentication flow...")
        print(f"  Instance: {instance_url}")
        print(f"  Callback: {redirect_uri}")
        print(f"\nOpening browser for GitLab authentication...")
        print(f"If the browser doesn't open, visit:\n  {auth_url}\n")

        # Start local callback server and open browser
        auth_code_future = asyncio.ensure_future(
            cls._run_callback_server(callback_port, state)
        )

        # Give server a moment to start, then open auth URL
        await asyncio.sleep(0.1)

        if auth_url_handler is not None:
            try:
                await auth_url_handler(auth_url)
            except Exception as e:
                auth_code_future.cancel()
                with suppress(asyncio.CancelledError):
                    await auth_code_future
                raise RuntimeError(
                    f"Failed to handle OAuth authorization URL: {e}"
                ) from e
        elif auto_open_browser:
            webbrowser.open(auth_url)

        auth_code = await auth_code_future

        if not auth_code:
            raise RuntimeError("OAuth flow failed: no authorization code received")

        print("Authorization code received. Exchanging for tokens...")

        # Exchange authorization code for tokens
        token_url = f"{instance_url}/oauth/token"
        token_payload = {
            "client_id": client_id,
            "code": auth_code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                token_url,
                data=token_payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15.0,
            )

            if response.status_code >= 400:
                raise RuntimeError(
                    f"Token exchange failed ({response.status_code}): {response.text[:500]}"
                )

            data = response.json()

        # Build credential object
        creds = {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token", ""),
            "token_type": data.get("token_type", "Bearer"),
            "expires_at": time.time() + data.get("expires_in", 7200),
            "instance_url": instance_url,
            "client_id": client_id,
            "created_at": time.time(),
        }

        # Determine output path
        if not output_path:
            oauth_dir = Path("oauth_creds")
            oauth_dir.mkdir(exist_ok=True)
            # Find next available slot
            idx = 1
            while (oauth_dir / f"gitlab_duo_oauth_{idx}.json").exists():
                idx += 1
            output_path = str(oauth_dir / f"gitlab_duo_oauth_{idx}.json")

        # Save credentials
        cls._save_oauth_credentials(output_path, creds)

        print(f"\nOAuth credentials saved to: {output_path}")
        print("You can now use this credential with the proxy.")
        return output_path

    @classmethod
    async def _run_callback_server(
        cls, port: int, expected_state: str
    ) -> Optional[str]:
        """
        Start a temporary HTTP server to receive the OAuth callback.

        Returns the authorization code from the callback, or None on failure.
        """
        auth_code: Optional[str] = None
        server_ready = asyncio.Event()
        code_received = asyncio.Event()

        async def handle_request(reader, writer):
            nonlocal auth_code
            try:
                request_line = await asyncio.wait_for(reader.readline(), timeout=30.0)
                request_text = request_line.decode("utf-8", errors="ignore")

                # Parse the GET request
                if request_text.startswith("GET "):
                    path = request_text.split(" ")[1]
                    parsed = urlparse(path)

                    if parsed.path == OAUTH_CALLBACK_PATH:
                        params = parse_qs(parsed.query)
                        received_state = params.get("state", [None])[0]
                        code = params.get("code", [None])[0]
                        error = params.get("error", [None])[0]

                        if error:
                            error_desc = params.get("error_description", [error])[0]
                            body = (
                                f"<html><body><h2>Authentication Failed</h2>"
                                f"<p>{error_desc}</p>"
                                f"<p>You can close this window.</p></body></html>"
                            )
                        elif received_state != expected_state:
                            body = (
                                "<html><body><h2>Authentication Failed</h2>"
                                "<p>Invalid state parameter (CSRF protection).</p>"
                                "<p>You can close this window.</p></body></html>"
                            )
                        elif code:
                            auth_code = code
                            body = (
                                "<html><body><h2>Authentication Successful!</h2>"
                                "<p>You can close this window and return to the terminal.</p>"
                                "</body></html>"
                            )
                        else:
                            body = (
                                "<html><body><h2>Authentication Failed</h2>"
                                "<p>No authorization code received.</p>"
                                "<p>You can close this window.</p></body></html>"
                            )

                        response = (
                            f"HTTP/1.1 200 OK\r\n"
                            f"Content-Type: text/html\r\n"
                            f"Content-Length: {len(body)}\r\n"
                            f"Connection: close\r\n\r\n"
                            f"{body}"
                        )
                        writer.write(response.encode())
                        await writer.drain()
                        code_received.set()
            except Exception as e:
                lib_logger.debug("[GitLabDuo] Callback handler error: %s", e)
            finally:
                writer.close()

        server = await asyncio.start_server(handle_request, "127.0.0.1", port)
        server_ready.set()

        try:
            # Wait for the callback (timeout after 5 minutes)
            await asyncio.wait_for(code_received.wait(), timeout=300.0)
        except asyncio.TimeoutError:
            print("\n[GitLabDuo OAuth] Timed out waiting for authentication.")
        finally:
            server.close()
            await server.wait_closed()

        return auth_code

    # =========================================================================
    # ACOMPLETION
    # =========================================================================

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Handle chat completion request for GitLab Duo.

        Routes to Anthropic or OpenAI proxy based on model type.
        """
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)

        # DEBUG: trace thinking params arriving at provider
        lib_logger.info(
            "[GitLabDuo] acompletion kwargs: thinking_budget=%s, reasoning_effort=%s, max_tokens=%s",
            kwargs.get("thinking_budget"),
            kwargs.get("reasoning_effort"),
            kwargs.get("max_tokens"),
        )

        api_key = kwargs.pop("credential_identifier", kwargs.pop("credential_path", ""))
        extra_headers = kwargs.pop("extra_headers", None)
        forwarded_headers: Dict[str, str] = {}
        if isinstance(extra_headers, dict):
            for key, value in extra_headers.items():
                if value is None:
                    continue
                key_l = str(key).lower().strip()
                if key_l in {"anthropic-beta", "anthropic-version"}:
                    forwarded_headers[key_l] = str(value)

        # Strip provider prefix
        clean_model = model.split("/", 1)[1] if "/" in model else model

        model_info = MODEL_MAP.get(clean_model)
        if not model_info:
            raise ValueError(f"Unknown GitLab Duo model: {clean_model}")

        backend_model, provider_type = model_info

        # Get direct access token (with cache)
        try:
            token_data = await self._get_or_refresh_token(api_key, client)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                self._invalidate_token(api_key)
            raise

        # Build headers from token data
        gateway_headers = dict(token_data.get("headers", {}))
        gateway_headers["Authorization"] = f"Bearer {token_data['token']}"
        gateway_headers["Content-Type"] = "application/json"

        ai_gateway_url = _get_ai_gateway_url()

        # Filter kwargs to pass through
        filtered = {
            k: v
            for k, v in kwargs.items()
            if k
            not in (
                "model",
                "messages",
                "stream",
                "credential_identifier",
                "credential_path",
                "extra_headers",
            )
        }

        if provider_type == "anthropic":
            result = await self._anthropic_completion(
                client=client,
                ai_gateway_url=ai_gateway_url,
                headers=gateway_headers,
                backend_model=backend_model,
                proxy_model=clean_model,
                messages=messages,
                stream=stream,
                api_key=api_key,
                forwarded_headers=forwarded_headers,
                **filtered,
            )
        else:
            result = await self._openai_completion(
                client=client,
                ai_gateway_url=ai_gateway_url,
                headers=gateway_headers,
                backend_model=backend_model,
                proxy_model=clean_model,
                messages=messages,
                stream=stream,
                api_key=api_key,
                **filtered,
            )

        # Track credit usage
        if stream:
            # Wrap the async generator to record credits on stream completion
            return self._wrap_stream_with_credit_tracking(result, api_key, clean_model)
        else:
            # Non-streaming: record immediately
            self.record_credit_usage(api_key, clean_model)
            return result

    # =========================================================================
    # STREAM CREDIT TRACKING WRAPPER
    # =========================================================================

    async def _wrap_stream_with_credit_tracking(
        self,
        stream: AsyncGenerator[litellm.ModelResponse, None],
        api_key: str,
        clean_model: str,
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """Wrap a streaming response to record credit usage on completion."""
        try:
            async for chunk in stream:
                yield chunk
        finally:
            # Record credit usage when stream completes (success or error)
            self.record_credit_usage(api_key, clean_model)

    # =========================================================================
    # OPENAI→ANTHROPIC MESSAGE CONVERSION
    # =========================================================================

    @staticmethod
    def _inject_interleaved_thinking_reminder(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Inject interleaved thinking reminder into the last user message.

        Appends a text block to the last user message that contains actual text
        (not just tool_result). This nudges Claude to emit thinking blocks on
        every response during tool-use loops.
        """
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") != "user":
                continue

            content = msg.get("content", [])
            # String content — has real text
            if isinstance(content, str):
                messages[i]["content"] = [
                    {"type": "text", "text": content},
                    {"type": "text", "text": INTERLEAVED_THINKING_REMINDER},
                ]
                return messages

            # List content — check for real text (not just tool_result)
            if isinstance(content, list):
                has_text = any(
                    isinstance(b, dict)
                    and b.get("type") == "text"
                    and b.get("text", "").strip()
                    for b in content
                )
                has_tool_result = any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in content
                )
                if has_text and not has_tool_result:
                    content.append(
                        {"type": "text", "text": INTERLEAVED_THINKING_REMINDER}
                    )
                    return messages

        return messages

    @staticmethod
    def _sanitize_tool_id(tool_id: str) -> str:
        """Sanitize tool ID to match Anthropic's pattern: ^[a-zA-Z0-9_-]+$"""
        if not tool_id:
            return f"toolu_{uuid.uuid4().hex[:12]}"
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_id)
        return sanitized if sanitized else f"toolu_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _merge_csv_header_values(*values: Optional[str]) -> Optional[str]:
        """Merge comma-separated header values while preserving order."""
        seen = set()
        merged: List[str] = []
        for value in values:
            if not value:
                continue
            for part in value.split(","):
                token = part.strip()
                if token and token not in seen:
                    seen.add(token)
                    merged.append(token)
        return ",".join(merged) if merged else None

    def _openai_to_anthropic_messages(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Optional[Union[str, List[Dict[str, Any]]]], List[Dict[str, Any]]]:
        """
        Convert OpenAI-format messages to Anthropic Messages API format.

        Returns:
            Tuple of (system_content, anthropic_messages)
        """
        system_parts: List[str] = []
        system_blocks: List[Dict[str, Any]] = []
        anthropic_msgs: List[Dict[str, Any]] = []

        def _ensure_role(role: str) -> Dict[str, Any]:
            """Get or create a message with the given role at the end."""
            if anthropic_msgs and anthropic_msgs[-1].get("role") == role:
                return anthropic_msgs[-1]
            msg: Dict[str, Any] = {"role": role, "content": []}
            anthropic_msgs.append(msg)
            return msg

        def _convert_image(url_data: str) -> Optional[Dict[str, Any]]:
            """Convert an image URL or data URI to Anthropic image block."""
            if url_data.startswith("data:"):
                media_part, _, b64_data = url_data.partition(";base64,")
                media_type = media_part.replace("data:", "") or "image/png"
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_data,
                    },
                }
            return {
                "type": "image",
                "source": {"type": "url", "url": url_data},
            }

        def _content_to_blocks(content: Any) -> List[Dict[str, Any]]:
            """Convert OpenAI content (str or list) to Anthropic blocks."""
            if isinstance(content, str):
                return [{"type": "text", "text": content}] if content else []
            if not isinstance(content, list):
                return [{"type": "text", "text": str(content)}] if content else []

            blocks: List[Dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type", "text")
                if ptype == "text":
                    text = part.get("text", "")
                    if text:
                        block: Dict[str, Any] = {"type": "text", "text": text}
                        # Pass through cache_control for prompt caching
                        if "cache_control" in part:
                            block["cache_control"] = part["cache_control"]
                        blocks.append(block)
                elif ptype == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url:
                        img = _convert_image(url)
                        if img:
                            # Pass through cache_control for images too
                            if "cache_control" in part:
                                img["cache_control"] = part["cache_control"]
                            blocks.append(img)
            return blocks

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # System messages → separate system parameter
            if role == "system":
                if isinstance(content, str):
                    if content:
                        system_parts.append(content)
                elif isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "text":
                            text = b.get("text", "")
                            if text:
                                block: Dict[str, Any] = {"type": "text", "text": text}
                                if "cache_control" in b:
                                    block["cache_control"] = b["cache_control"]
                                system_blocks.append(block)
                continue

            # Tool results → user message with tool_result blocks
            if role == "tool":
                user_msg = _ensure_role("user")
                tool_content = content
                if isinstance(tool_content, list):
                    converted_tool_content = _content_to_blocks(tool_content)
                    tool_content = (
                        converted_tool_content
                        if converted_tool_content
                        else json.dumps(tool_content)
                    )
                elif not isinstance(tool_content, str):
                    tool_content = str(tool_content) if tool_content else ""
                user_msg["content"].append(
                    {
                        "type": "tool_result",
                        "tool_use_id": self._sanitize_tool_id(
                            msg.get("tool_call_id", "")
                        ),
                        "content": tool_content,
                    }
                )
                continue

            # Assistant messages
            if role == "assistant":
                asst_msg = _ensure_role("assistant")
                blocks = asst_msg["content"]

                # Thinking / reasoning – only include if we have a valid signature.
                # Anthropic rejects thinking blocks with missing/invalid signatures.
                reasoning = msg.get("reasoning_content") or msg.get("reasoning")
                sig = msg.get("thinking_signature", "")
                if reasoning and sig and len(sig) >= 50:
                    blocks.append(
                        {
                            "type": "thinking",
                            "thinking": reasoning,
                            "signature": sig,
                        }
                    )

                # Text content
                text_blocks = _content_to_blocks(content)
                blocks.extend(text_blocks)

                # Tool calls → tool_use blocks
                for tc in msg.get("tool_calls", []):
                    func = tc.get("function", {})
                    # Anthropic requires name to have ≥1 character
                    fname = func.get("name") or tc.get("name") or "_unknown"
                    try:
                        input_data = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        input_data = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": self._sanitize_tool_id(tc.get("id", "")),
                            "name": fname,
                            "input": input_data,
                        }
                    )
                continue

            # User messages
            if role == "user":
                user_msg = _ensure_role("user")
                user_msg["content"].extend(_content_to_blocks(content))

        system_payload: Optional[Union[str, List[Dict[str, Any]]]] = None
        if system_blocks:
            if system_parts:
                system_blocks.insert(
                    0, {"type": "text", "text": "\n\n".join(system_parts)}
                )
            system_payload = system_blocks
        elif system_parts:
            system_payload = "\n\n".join(system_parts)

        # Fix orphaned tool_use blocks that have no matching tool_result
        anthropic_msgs = self._fix_orphaned_tool_use(anthropic_msgs)

        return system_payload, anthropic_msgs

    @staticmethod
    def _fix_orphaned_tool_use(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Ensure every assistant tool_use block has a matching tool_result
        in the immediately following user message.

        Anthropic requires that each tool_use in an assistant message is
        paired with a tool_result in the very next (user) message. Clients
        (especially mobile apps with web-search) sometimes send truncated
        histories where tool results are missing. This method:

        1. Collects tool_use IDs from each assistant message.
        2. Checks the next message for matching tool_result blocks.
        3. For any orphaned tool_use IDs, either:
           a. Inserts a synthetic user message with placeholder tool_results
              (if the next message isn't a user message or doesn't exist), or
           b. Appends the missing tool_result blocks to the existing next
              user message.

        This prevents Anthropic 400 errors like:
        "tool_use ids were found without tool_result blocks immediately after"
        """
        if not messages:
            return messages

        result: List[Dict[str, Any]] = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            result.append(msg)

            if msg.get("role") != "assistant":
                i += 1
                continue

            # Collect all tool_use IDs in this assistant message
            tool_use_ids: List[str] = []
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tid = block.get("id")
                    if tid:
                        tool_use_ids.append(tid)

            if not tool_use_ids:
                i += 1
                continue

            # Check what's in the next message
            next_msg = messages[i + 1] if i + 1 < len(messages) else None

            if next_msg and next_msg.get("role") == "user":
                # Find which tool_use IDs already have results
                existing_result_ids = set()
                for block in next_msg.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        rid = block.get("tool_use_id")
                        if rid:
                            existing_result_ids.add(rid)

                # Add placeholder results for any missing ones
                missing_ids = [
                    tid for tid in tool_use_ids if tid not in existing_result_ids
                ]
                if missing_ids:
                    lib_logger.warning(
                        "[GitLabDuo] Injecting %d placeholder tool_result(s) "
                        "for orphaned tool_use IDs: %s",
                        len(missing_ids),
                        missing_ids,
                    )
                    for tid in missing_ids:
                        next_msg["content"].insert(
                            0,
                            {
                                "type": "tool_result",
                                "tool_use_id": tid,
                                "content": "[Tool result not available]",
                            },
                        )
            else:
                # Next message is another assistant or doesn't exist —
                # insert a synthetic user message with all tool_results
                lib_logger.warning(
                    "[GitLabDuo] Inserting synthetic tool_result message "
                    "for %d orphaned tool_use IDs: %s",
                    len(tool_use_ids),
                    tool_use_ids,
                )
                synthetic_user: Dict[str, Any] = {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tid,
                            "content": "[Tool result not available]",
                        }
                        for tid in tool_use_ids
                    ],
                }
                result.append(synthetic_user)

            i += 1

        return result

    def _openai_tools_to_anthropic(
        self, tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert OpenAI tools to Anthropic format."""
        if not tools:
            return None
        result = []
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                continue
            func = tool.get("function", {})
            name = func.get("name")
            if not name:
                continue
            converted = {
                "name": name,
                "description": func.get("description", ""),
                "input_schema": func.get("parameters")
                or {"type": "object", "properties": {}},
            }
            if "cache_control" in tool:
                converted["cache_control"] = tool["cache_control"]
            result.append(converted)
        return result or None

    def _openai_tool_choice_to_anthropic(
        self, tool_choice: Any
    ) -> Optional[Dict[str, Any]]:
        """Convert OpenAI tool_choice to Anthropic format."""
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            mapping = {
                "auto": {"type": "auto"},
                "required": {"type": "any"},
                "any": {"type": "any"},
                "none": {"type": "none"},
            }
            return mapping.get(tool_choice.strip().lower(), {"type": "auto"})
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            func = tool_choice.get("function", {})
            if func.get("name"):
                return {"type": "tool", "name": func["name"]}
        return {"type": "auto"}

    @staticmethod
    def _resolve_effort_level(kwargs: Dict[str, Any]) -> str:
        """Resolve the adaptive thinking effort level from request kwargs.

        Priority:
        1. Explicit disable (reasoning_effort=disable/off/none) -> returns "off"
        2. output_config.effort passed via Anthropic SDK path
        3. thinking_budget (converted via BUDGET_TO_EFFORT_THRESHOLDS)
        4. reasoning_effort (converted via REASONING_TO_EFFORT_MAP)
        5. Default "high"
        """
        # Check for explicit disable
        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort:
            effort_str = str(reasoning_effort).strip().lower()
            if effort_str in ("disable", "off", "none"):
                return "off"

        # Explicit effort from Anthropic SDK output_config
        effort = kwargs.get("effort")
        if effort:
            return str(effort).strip().lower()

        # Convert legacy budget_tokens to effort level
        thinking_budget = kwargs.get("thinking_budget")
        if thinking_budget is not None:
            budget = int(thinking_budget)
            for threshold, level in BUDGET_TO_EFFORT_THRESHOLDS:
                if budget <= threshold:
                    return level
            return "max"

        # Convert reasoning_effort name to effort level
        if reasoning_effort:
            return REASONING_TO_EFFORT_MAP.get(
                str(reasoning_effort).strip().lower(), "high"
            )

        # Default
        return "high"

    # =========================================================================
    # ANTHROPIC (CLAUDE) COMPLETION
    # =========================================================================

    async def _anthropic_completion(
        self,
        client: httpx.AsyncClient,
        ai_gateway_url: str,
        headers: Dict[str, str],
        backend_model: str,
        proxy_model: str,
        messages: List[Dict[str, Any]],
        stream: bool,
        api_key: str = "",
        forwarded_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """Handle completion via Anthropic Messages API proxy."""
        system_content, anthropic_messages = self._openai_to_anthropic_messages(
            messages
        )

        payload: Dict[str, Any] = {
            "model": backend_model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens") or 8192,
        }

        if system_content:
            # Auto-add cache_control for long system prompts (>4000 chars ≈ 1024 tokens)
            # only when no explicit cache_control markers are provided.
            if isinstance(system_content, str):
                if len(system_content) > 4000:
                    payload["system"] = [
                        {
                            "type": "text",
                            "text": system_content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                else:
                    payload["system"] = system_content
            elif isinstance(system_content, list):
                payload["system"] = system_content
                has_explicit_cache = any(
                    isinstance(block, dict) and "cache_control" in block
                    for block in system_content
                )
                if not has_explicit_cache:
                    total_system_chars = sum(
                        len(block.get("text", ""))
                        for block in system_content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                    if total_system_chars > 4000:
                        for i in range(len(payload["system"]) - 1, -1, -1):
                            block = payload["system"][i]
                            if (
                                isinstance(block, dict)
                                and block.get("type") == "text"
                                and block.get("text", "").strip()
                            ):
                                block["cache_control"] = {"type": "ephemeral"}
                                break
        if stream:
            payload["stream"] = True

        # Tools
        tools = self._openai_tools_to_anthropic(kwargs.get("tools"))
        if tools:
            payload["tools"] = tools
            tc = self._openai_tool_choice_to_anthropic(kwargs.get("tool_choice"))
            if tc:
                payload["tool_choice"] = tc

        # Thinking / reasoning
        thinking_budget = kwargs.get("thinking_budget")
        thinking_type = kwargs.get("thinking_type")  # "enabled", "adaptive", etc.
        reasoning_effort = kwargs.get("reasoning_effort")
        enable_thinking = False
        use_adaptive = backend_model in ADAPTIVE_THINKING_MODELS

        if use_adaptive:
            # --- 4.6 models: adaptive thinking with effort levels ---
            effort = self._resolve_effort_level(kwargs)
            if effort == "off":
                payload["thinking"] = {"type": "disabled"}
            else:
                payload["thinking"] = {"type": "adaptive"}
                payload["output_config"] = {"effort": effort}
                enable_thinking = True
                # Floor max_tokens so the model has room to think
                if payload["max_tokens"] < 16384:
                    payload["max_tokens"] = 16384
                lib_logger.info(
                    "[GitLabDuo] Adaptive thinking: effort=%s, max_tokens=%d, model=%s",
                    effort,
                    payload["max_tokens"],
                    backend_model,
                )
        else:
            # --- Legacy models (4.5, haiku): budget_tokens thinking ---
            if thinking_budget is not None:
                # Explicit budget from client
                budget = int(thinking_budget)
                payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
                enable_thinking = True
            elif reasoning_effort:
                effort = str(reasoning_effort).strip().lower()
                if effort not in ("disable", "off", "none"):
                    budget = THINKING_BUDGET_MAP.get(effort, 31999)
                    payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
                    enable_thinking = True
            elif thinking_type and thinking_type != "disabled":
                # Client requested thinking without explicit budget
                budget = 31999
                payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
                enable_thinking = True
            else:
                # Enable by default for opus/sonnet, but NOT for haiku
                if "haiku" not in backend_model.lower():
                    budget = THINKING_BUDGET_MAP.get("auto", 31999)
                    payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
                    enable_thinking = True

            # Ensure max_tokens > budget_tokens when thinking is enabled
            if enable_thinking:
                budget = payload["thinking"]["budget_tokens"]
                if payload["max_tokens"] <= budget:
                    payload["max_tokens"] = budget + max(payload["max_tokens"], 8192)
                lib_logger.info(
                    "[GitLabDuo] Thinking enabled: budget_tokens=%d, max_tokens=%d, model=%s",
                    budget,
                    payload["max_tokens"],
                    backend_model,
                )

        # Optional parameters
        if kwargs.get("temperature") is not None:
            payload["temperature"] = kwargs["temperature"]
        if kwargs.get("top_p") is not None:
            payload["top_p"] = kwargs["top_p"]
        stop = kwargs.get("stop")
        if stop:
            payload["stop_sequences"] = stop if isinstance(stop, list) else [stop]

        # Inject interleaved thinking reminder when thinking + tools are active
        if enable_thinking and tools:
            payload["messages"] = self._inject_interleaved_thinking_reminder(
                payload["messages"]
            )

        endpoint = f"{ai_gateway_url}/ai/v1/proxy/anthropic/v1/messages"

        # Anthropic-specific headers
        forwarded_headers = forwarded_headers or {}
        req_headers = dict(headers)
        req_headers["anthropic-version"] = str(
            forwarded_headers.get("anthropic-version") or ANTHROPIC_VERSION
        )

        incoming_beta = forwarded_headers.get("anthropic-beta")
        # 4.6 adaptive models auto-enable interleaved thinking; skip the beta header
        thinking_beta = (
            ANTHROPIC_BETA if enable_thinking and not use_adaptive else None
        )
        merged_beta = self._merge_csv_header_values(incoming_beta, thinking_beta)
        if merged_beta:
            req_headers["anthropic-beta"] = merged_beta

        lib_logger.debug(
            "[GitLabDuo] Anthropic request to %s: %s...",
            backend_model,
            json.dumps(payload, default=str)[:500],
        )

        if stream:
            return self._stream_anthropic_with_retry(
                client, endpoint, req_headers, payload, proxy_model, api_key
            )
        else:
            return await self._non_stream_anthropic_response(
                client, endpoint, req_headers, payload, proxy_model, api_key
            )

    async def _non_stream_anthropic_response(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        proxy_model: str,
        api_key: str = "",
    ) -> litellm.ModelResponse:
        """Handle non-streaming Anthropic Messages response."""
        response = await client.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=TimeoutConfig.non_streaming(),
        )

        if response.status_code == 401:
            self._invalidate_token(api_key)

        if response.status_code >= 400:
            error_text = response.text[:500]
            lib_logger.error(
                "[GitLabDuo] Anthropic API error %d: %s",
                response.status_code,
                error_text,
            )
            # 403 Forbidden — account banned/suspended, remove immediately
            if response.status_code == 403:
                self._handle_credential_forbidden(api_key)

            # Track 402 credit exhaustion strikes
            if response.status_code == 402 and self._is_credit_exhaustion(
                response.text
            ):
                strikes = self._record_exhaustion_strike(api_key)
                if strikes >= MAX_EXHAUSTION_STRIKES:
                    self._removed_credentials.add(api_key)

            raise httpx.HTTPStatusError(
                f"GitLab Duo API error: {response.status_code}",
                request=response.request,
                response=response,
            )

        data = response.json()
        return self._anthropic_response_to_litellm(data, proxy_model)

    def _anthropic_response_to_litellm(
        self, data: Dict[str, Any], proxy_model: str
    ) -> litellm.ModelResponse:
        """Convert Anthropic Messages response to litellm.ModelResponse."""
        content_text = ""
        reasoning_content = ""
        thinking_signature = ""
        tool_calls = []

        for block in data.get("content", []):
            btype = block.get("type", "")
            if btype == "text":
                content_text += block.get("text", "")
            elif btype == "thinking":
                reasoning_content += block.get("thinking", "")
                if block.get("signature"):
                    thinking_signature = block["signature"]
            elif btype == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    }
                )

        # Map stop_reason → finish_reason
        stop_reason = data.get("stop_reason", "end_turn")
        finish_map = {
            "end_turn": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
            "stop_sequence": "stop",
        }
        finish_reason = finish_map.get(stop_reason, "stop")

        message: Dict[str, Any] = {
            "role": "assistant",
            "content": content_text or None,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        if reasoning_content:
            message["reasoning_content"] = reasoning_content
            if thinking_signature:
                message["thinking_signature"] = thinking_signature

        response_obj = litellm.ModelResponse(
            id=data.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
            created=int(time.time()),
            model=f"gitlab_duo/{proxy_model}",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
        )

        usage = data.get("usage", {})
        if usage:
            inp = usage.get("input_tokens", 0)
            out = usage.get("output_tokens", 0)
            cache_creation = usage.get("cache_creation_input_tokens", 0)
            cache_read = usage.get("cache_read_input_tokens", 0)

            usage_obj = litellm.Usage(
                prompt_tokens=inp,
                completion_tokens=out,
                total_tokens=inp + out,
            )
            # Attach cache token info via prompt_tokens_details (survives model_dump)
            # and also as direct attributes for legacy consumers
            if cache_creation or cache_read:
                usage_obj.prompt_tokens_details = {
                    "cached_tokens": cache_read or 0,
                    "cache_creation_tokens": cache_creation or 0,
                }
                usage_obj.cache_creation_input_tokens = cache_creation
                usage_obj.cache_read_input_tokens = cache_read
                lib_logger.debug(
                    "[GitLabDuo] Cache tokens: creation=%d, read=%d",
                    cache_creation,
                    cache_read,
                )
            response_obj.usage = usage_obj

        return response_obj

    # Maximum internal retries for transient stream failures
    _STREAM_TRANSIENT_MAX_RETRIES = 3
    _TRANSIENT_402_DELAY = 5  # seconds
    _TRANSIENT_OVERLOADED_DELAY = 3  # seconds

    @staticmethod
    def _is_credit_exhaustion(error_body: str) -> bool:
        """Check if a 402 error body indicates real credit exhaustion vs transient."""
        body_lower = error_body.lower()
        return any(
            marker in body_lower
            for marker in (
                "insufficient_credits",
                "usage_quota_exceeded",
                "credit",
                "quota exceeded",
                "duo credits",
            )
        )

    @staticmethod
    def _is_overloaded_stream_error(error_body: str) -> bool:
        """Check whether a stream failure body indicates transient overload."""
        body_lower = (error_body or "").lower()
        return (
            "overloaded_error" in body_lower
            or '"message": "overloaded"' in body_lower
            or "model_capacity_exhausted" in body_lower
        )

    @staticmethod
    def _parse_retry_after_seconds(header_value: Optional[str]) -> Optional[float]:
        """Parse Retry-After header value if present."""
        if not header_value:
            return None
        try:
            value = float(str(header_value).strip())
            if value > 0:
                return value
        except (TypeError, ValueError):
            return None
        return None

    def _record_exhaustion_strike(self, api_key: str) -> int:
        """Record an exhaustion strike for a credential. Returns new strike count.

        When strikes reach ``MAX_EXHAUSTION_STRIKES``:
        - The credential file is renamed to ``*.exhausted`` so it is not
          re-discovered on restart.
        - The registered ``_credential_exhausted_callback`` (if any) is
          scheduled via ``asyncio.create_task``.
        """
        strikes = self._exhaustion_strikes.get(api_key, 0) + 1
        self._exhaustion_strikes[api_key] = strikes
        masked = api_key[-8:] if len(api_key) > 8 else api_key[:4]
        lib_logger.warning(
            "[GitLabDuo] Credit exhaustion strike %d/%d for credential ...%s",
            strikes,
            MAX_EXHAUSTION_STRIKES,
            masked,
        )

        # Handle threshold reached — fire exactly once
        if strikes == MAX_EXHAUSTION_STRIKES:
            # Retire the credential file so it is not loaded on next restart
            self._retire_credential_file(api_key)

            # Fire the optional callback (auto-newaccount / Telegram notification)
            if self._credential_exhausted_callback is not None:
                try:
                    lib_logger.info(
                        "[GitLabDuo] Credential ...%s reached %d strikes — "
                        "scheduling auto-newaccount callback",
                        masked,
                        strikes,
                    )
                    asyncio.create_task(self._credential_exhausted_callback(api_key))
                except Exception:
                    lib_logger.exception(
                        "[GitLabDuo] Failed to schedule credential exhausted callback for ...%s",
                        masked,
                    )

        return strikes

    @staticmethod
    def _retire_credential_file(api_key: str) -> None:
        """Rename an exhausted credential file to ``*.exhausted`` so it is
        not re-discovered by the ``*_oauth_*.json`` glob on restart.

        Only operates on file-based credentials (OAuth JSON files).
        PAT strings are silently ignored.
        """
        cred_path = Path(api_key)
        if not cred_path.is_file():
            return
        retired_path = cred_path.with_suffix(".json.exhausted")
        try:
            cred_path.rename(retired_path)
            lib_logger.info(
                "[GitLabDuo] Retired exhausted credential: %s → %s",
                cred_path.name,
                retired_path.name,
            )
        except OSError as exc:
            lib_logger.error(
                "[GitLabDuo] Failed to retire credential file %s: %s",
                cred_path.name,
                exc,
            )

    def _handle_credential_forbidden(self, api_key: str) -> None:
        """Immediately remove a credential that returned 403 Forbidden.

        Unlike exhaustion (which needs 3 strikes), a 403 means the account
        is banned, suspended, or access-revoked — no point retrying.
        The credential file is retired and the exhaustion callback is fired
        so a replacement can be queued.
        """
        if api_key in self._removed_credentials:
            return  # Already handled

        masked = api_key[-8:] if len(api_key) > 8 else api_key[:4]
        lib_logger.error(
            "[GitLabDuo] Credential ...%s returned 403 Forbidden — "
            "removing immediately",
            masked,
        )
        self._removed_credentials.add(api_key)
        self._retire_credential_file(api_key)

        if self._credential_exhausted_callback is not None:
            try:
                asyncio.create_task(self._credential_exhausted_callback(api_key))
            except Exception:
                lib_logger.exception(
                    "[GitLabDuo] Failed to schedule callback for "
                    "forbidden credential ...%s",
                    masked,
                )

    async def _stream_anthropic_with_retry(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        proxy_model: str,
        api_key: str = "",
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """Wrap _stream_anthropic_response with internal transient retries.

        Distinguishes between:
        - Real credit exhaustion (insufficient_credits): increment strikes, propagate
          immediately so the executor rotates to the next credential
        - Transient 402: retry briefly, then propagate
        - Transient overload (503/429): retry briefly, then propagate
        """
        import asyncio as _asyncio

        for attempt in range(self._STREAM_TRANSIENT_MAX_RETRIES):
            try:
                async for chunk in self._stream_anthropic_response(
                    client, endpoint, headers, payload, proxy_model, api_key
                ):
                    yield chunk
                return  # Success — stream completed
            except httpx.HTTPStatusError as e:
                status = e.response.status_code

                # 403 Forbidden — immediate removal before propagating
                if status == 403:
                    self._handle_credential_forbidden(api_key)
                    raise

                error_body = ""
                try:
                    error_body = e.response.text or ""
                except Exception:
                    try:
                        error_body = e.response.content.decode("utf-8", errors="ignore")
                    except Exception:
                        pass

                # Handle overload and temporary capacity errors.
                if status in (429, 503) and self._is_overloaded_stream_error(
                    error_body
                ):
                    if attempt < self._STREAM_TRANSIENT_MAX_RETRIES - 1:
                        retry_after = self._parse_retry_after_seconds(
                            e.response.headers.get("retry-after")
                        )
                        delay = retry_after or self._TRANSIENT_OVERLOADED_DELAY
                        lib_logger.info(
                            "[GitLabDuo] Transient overload (%d), retrying in %.1fs "
                            "(attempt %d/%d)",
                            status,
                            delay,
                            attempt + 1,
                            self._STREAM_TRANSIENT_MAX_RETRIES,
                        )
                        await _asyncio.sleep(delay)
                        continue
                    raise  # Last attempt — propagate

                if status != 402:
                    raise  # Non-402 — propagate immediately

                if self._is_credit_exhaustion(error_body):
                    # Real credit exhaustion — record strike and propagate
                    # The executor will rotate to the next credential
                    strikes = self._record_exhaustion_strike(api_key)
                    if strikes >= MAX_EXHAUSTION_STRIKES:
                        self._removed_credentials.add(api_key)
                        lib_logger.error(
                            "[GitLabDuo] Credential reached %d exhaustion strikes, "
                            "marked for removal: ...%s",
                            strikes,
                            api_key[-8:] if len(api_key) > 8 else api_key[:4],
                        )
                    raise  # Always propagate — let executor handle rotation

                # Transient 402 — retry with delay
                if attempt < self._STREAM_TRANSIENT_MAX_RETRIES - 1:
                    lib_logger.info(
                        "[GitLabDuo] Transient 402, retrying in %ds (attempt %d/%d)",
                        self._TRANSIENT_402_DELAY,
                        attempt + 1,
                        self._STREAM_TRANSIENT_MAX_RETRIES,
                    )
                    await _asyncio.sleep(self._TRANSIENT_402_DELAY)
                    continue
                raise  # Last attempt — propagate

    async def _stream_anthropic_response(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        proxy_model: str,
        api_key: str = "",
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """
        Stream Anthropic Messages API response → litellm chunks.

        Parses Anthropic SSE events (message_start, content_block_start,
        content_block_delta, content_block_stop, message_delta, message_stop)
        and yields OpenAI-compatible litellm.ModelResponse chunks.
        """
        created = int(time.time())
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        is_first_chunk = True

        # Tool call accumulation state
        current_tool_calls: Dict[int, Dict[str, Any]] = {}
        current_block_type: Optional[str] = None

        stream_headers = {**headers, "Accept-Encoding": "identity"}

        async with client.stream(
            "POST",
            endpoint,
            headers=stream_headers,
            json=payload,
            timeout=TimeoutConfig.streaming(),
        ) as response:
            if response.status_code == 401:
                self._invalidate_token(api_key)

            if response.status_code >= 400:
                error_body = await response.aread()
                error_text = error_body.decode("utf-8", errors="ignore")
                lib_logger.error(
                    "[GitLabDuo] Anthropic stream error %d: %s",
                    response.status_code,
                    error_text[:500],
                )
                # Build a synthetic non-streaming response so classify_error
                # can read status_code and body even after the stream context exits.
                from httpx import Response as _Resp

                synth_resp = _Resp(
                    status_code=response.status_code,
                    headers=response.headers,
                    content=error_body,
                    request=response.request,
                )
                raise httpx.HTTPStatusError(
                    f"GitLab Duo API error: {response.status_code}",
                    request=response.request,
                    response=synth_resp,
                )

            captured_headers = {k.lower(): v for k, v in response.headers.items()}
            event_type: Optional[str] = None

            async for line in response.aiter_lines():
                if not line:
                    continue

                # Parse SSE event type
                if line.startswith("event: "):
                    event_type = line[7:].strip()
                    continue

                if not line.startswith("data: "):
                    continue

                data_str = line[6:].strip()
                if not data_str:
                    continue

                try:
                    evt = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if event_type == "message_start":
                    msg = evt.get("message", {})
                    if msg.get("id"):
                        response_id = msg["id"]

                elif event_type == "content_block_start":
                    block = evt.get("content_block", {})
                    current_block_type = block.get("type", "text")
                    block_index = evt.get("index", 0)
                    if current_block_type == "tool_use":
                        current_tool_calls[block_index] = {
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": "",
                            },
                        }

                elif event_type == "content_block_delta":
                    delta = evt.get("delta", {})
                    delta_type = delta.get("type", "")
                    block_index = evt.get("index", 0)

                    if delta_type == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            chunk = litellm.ModelResponse(
                                id=response_id,
                                created=created,
                                model=f"gitlab_duo/{proxy_model}",
                                object="chat.completion.chunk",
                                choices=[
                                    {
                                        "index": 0,
                                        "delta": {
                                            "content": text,
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

                    elif delta_type == "thinking_delta":
                        thinking = delta.get("thinking", "")
                        if thinking:
                            chunk = litellm.ModelResponse(
                                id=response_id,
                                created=created,
                                model=f"gitlab_duo/{proxy_model}",
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

                    elif delta_type == "signature_delta":
                        sig = delta.get("signature", "")
                        if sig:
                            chunk = litellm.ModelResponse(
                                id=response_id,
                                created=created,
                                model=f"gitlab_duo/{proxy_model}",
                                object="chat.completion.chunk",
                                choices=[
                                    {
                                        "index": 0,
                                        "delta": {
                                            "thinking_signature": sig,
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

                    elif delta_type == "input_json_delta":
                        partial = delta.get("partial_json", "")
                        if partial and block_index in current_tool_calls:
                            current_tool_calls[block_index]["function"][
                                "arguments"
                            ] += partial

                elif event_type == "message_delta":
                    delta = evt.get("delta", {})
                    stop_reason = delta.get("stop_reason", "end_turn")

                    finish_map = {
                        "end_turn": "stop",
                        "max_tokens": "length",
                        "tool_use": "tool_calls",
                        "stop_sequence": "stop",
                    }
                    finish_reason = finish_map.get(stop_reason, "stop")

                    # Emit accumulated tool calls before final chunk
                    if current_tool_calls and finish_reason == "tool_calls":
                        tool_calls_list = [
                            {"index": i, **current_tool_calls[i]}
                            for i in sorted(current_tool_calls.keys())
                        ]
                        chunk = litellm.ModelResponse(
                            id=response_id,
                            created=created,
                            model=f"gitlab_duo/{proxy_model}",
                            object="chat.completion.chunk",
                            choices=[
                                {
                                    "index": 0,
                                    "delta": {"tool_calls": tool_calls_list},
                                    "finish_reason": None,
                                }
                            ],
                        )
                        if is_first_chunk:
                            chunk._response_headers = captured_headers
                            is_first_chunk = False
                        yield chunk

                    # Final chunk with finish_reason
                    final_chunk = litellm.ModelResponse(
                        id=response_id,
                        created=created,
                        model=f"gitlab_duo/{proxy_model}",
                        object="chat.completion.chunk",
                        choices=[
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish_reason,
                            }
                        ],
                    )

                    usage = evt.get("usage", {})
                    if usage:
                        inp = usage.get("input_tokens", 0)
                        out = usage.get("output_tokens", 0)
                        cache_creation = usage.get("cache_creation_input_tokens", 0)
                        cache_read = usage.get("cache_read_input_tokens", 0)

                        usage_obj = litellm.Usage(
                            prompt_tokens=inp,
                            completion_tokens=out,
                            total_tokens=inp + out,
                        )
                        # Attach cache via prompt_tokens_details (survives model_dump)
                        if cache_creation or cache_read:
                            usage_obj.prompt_tokens_details = {
                                "cached_tokens": cache_read or 0,
                                "cache_creation_tokens": cache_creation or 0,
                            }
                            usage_obj.cache_creation_input_tokens = cache_creation
                            usage_obj.cache_read_input_tokens = cache_read
                        final_chunk.usage = usage_obj

                    if is_first_chunk:
                        final_chunk._response_headers = captured_headers
                        is_first_chunk = False
                    yield final_chunk

                elif event_type == "error":
                    err = evt.get("error") if isinstance(evt, dict) else None
                    err_type = ""
                    if isinstance(err, dict):
                        err_type = str(err.get("type", "")).lower()

                    if err_type in {"overloaded_error", "rate_limit_error"}:
                        from httpx import Response as _Resp

                        status_code = 503 if err_type == "overloaded_error" else 429
                        retry_after = None
                        if isinstance(err, dict):
                            retry_after = err.get("retry_after") or err.get(
                                "retry-after"
                            )

                        raw_body = (
                            evt
                            if isinstance(evt, dict)
                            else {"error": err or {"message": str(evt)}}
                        )
                        body_bytes = json.dumps(raw_body).encode("utf-8")
                        synth_headers = dict(response.headers)
                        if retry_after is not None:
                            synth_headers["retry-after"] = str(retry_after)

                        synth_resp = _Resp(
                            status_code=status_code,
                            headers=synth_headers,
                            content=body_bytes,
                            request=response.request,
                        )
                        raise httpx.HTTPStatusError(
                            f"GitLab Duo Anthropic stream error: {err or evt}",
                            request=response.request,
                            response=synth_resp,
                        )

                    raise RuntimeError(
                        f"GitLab Duo Anthropic stream error: {err or evt}"
                    )

    # =========================================================================
    # OPENAI (GPT) COMPLETION
    # =========================================================================

    async def _openai_completion(
        self,
        client: httpx.AsyncClient,
        ai_gateway_url: str,
        headers: Dict[str, str],
        backend_model: str,
        proxy_model: str,
        messages: List[Dict[str, Any]],
        stream: bool,
        api_key: str = "",
        **kwargs,
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """Handle completion via OpenAI Chat Completions proxy."""
        endpoint = f"{ai_gateway_url}/ai/v1/proxy/openai/v1/chat/completions"

        payload: Dict[str, Any] = {
            "model": backend_model,
            "messages": messages,
        }
        if stream:
            payload["stream"] = True

        # Pass through optional parameters
        for param in (
            "max_tokens",
            "temperature",
            "top_p",
            "stop",
            "tools",
            "tool_choice",
            "response_format",
            "reasoning_effort",
        ):
            if param in kwargs and kwargs[param] is not None:
                payload[param] = kwargs[param]

        lib_logger.debug(
            "[GitLabDuo] OpenAI request to %s: %s...",
            backend_model,
            json.dumps(payload, default=str)[:500],
        )

        if stream:
            return self._stream_openai_response(
                client, endpoint, headers, payload, proxy_model, api_key
            )
        else:
            return await self._non_stream_openai_response(
                client, endpoint, headers, payload, proxy_model, api_key
            )

    async def _non_stream_openai_response(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        proxy_model: str,
        api_key: str = "",
    ) -> litellm.ModelResponse:
        """Handle non-streaming OpenAI response."""
        response = await client.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=TimeoutConfig.non_streaming(),
        )

        if response.status_code == 401:
            self._invalidate_token(api_key)

        if response.status_code >= 400:
            lib_logger.error(
                "[GitLabDuo] OpenAI API error %d: %s",
                response.status_code,
                response.text[:500],
            )
            # 403 Forbidden — account banned/suspended, remove immediately
            if response.status_code == 403:
                self._handle_credential_forbidden(api_key)

            raise httpx.HTTPStatusError(
                f"GitLab Duo API error: {response.status_code}",
                request=response.request,
                response=response,
            )

        data = response.json()

        choices = []
        for choice in data.get("choices", []):
            message = choice.get("message", {})
            msg_dict: Dict[str, Any] = {
                "role": message.get("role", "assistant"),
                "content": message.get("content"),
            }
            if "tool_calls" in message:
                msg_dict["tool_calls"] = message["tool_calls"]
            if "reasoning_content" in message:
                msg_dict["reasoning_content"] = message["reasoning_content"]

            choices.append(
                {
                    "index": choice.get("index", 0),
                    "message": msg_dict,
                    "finish_reason": choice.get("finish_reason", "stop"),
                }
            )

        response_obj = litellm.ModelResponse(
            id=data.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}"),
            created=data.get("created", int(time.time())),
            model=f"gitlab_duo/{proxy_model}",
            object="chat.completion",
            choices=choices,
        )

        usage = data.get("usage", {})
        if usage:
            response_obj.usage = litellm.Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )

        return response_obj

    async def _stream_openai_response(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        proxy_model: str,
        api_key: str = "",
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """Stream OpenAI-format response and yield litellm chunks."""
        created = int(time.time())
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        is_first_chunk = True
        current_tool_calls: Dict[int, Dict[str, Any]] = {}

        stream_headers = {**headers, "Accept-Encoding": "identity"}

        async with client.stream(
            "POST",
            endpoint,
            headers=stream_headers,
            json=payload,
            timeout=TimeoutConfig.streaming(),
        ) as response:
            if response.status_code == 401:
                self._invalidate_token(api_key)

            if response.status_code >= 400:
                error_body = await response.aread()
                error_text = error_body.decode("utf-8", errors="ignore")
                lib_logger.error(
                    "[GitLabDuo] OpenAI stream error %d: %s",
                    response.status_code,
                    error_text[:500],
                )
                # 403 Forbidden — account banned/suspended, remove immediately
                if response.status_code == 403:
                    self._handle_credential_forbidden(api_key)

                from httpx import Response as _Resp

                synth_resp = _Resp(
                    status_code=response.status_code,
                    headers=response.headers,
                    content=error_body,
                    request=response.request,
                )
                raise httpx.HTTPStatusError(
                    f"GitLab Duo API error: {response.status_code}",
                    request=response.request,
                    response=synth_resp,
                )

            captured_headers = {k.lower(): v for k, v in response.headers.items()}

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data = line[6:].strip()
                if not data or data == "[DONE]":
                    continue

                try:
                    evt = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if evt.get("id"):
                    response_id = evt["id"]

                for choice in evt.get("choices", []):
                    index = choice.get("index", 0)
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason")

                    # Text content
                    if "content" in delta:
                        chunk = litellm.ModelResponse(
                            id=response_id,
                            created=created,
                            model=f"gitlab_duo/{proxy_model}",
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
                        if is_first_chunk:
                            chunk._response_headers = captured_headers
                            is_first_chunk = False
                        yield chunk

                    # Reasoning content
                    if "reasoning_content" in delta:
                        chunk = litellm.ModelResponse(
                            id=response_id,
                            created=created,
                            model=f"gitlab_duo/{proxy_model}",
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
                        if is_first_chunk:
                            chunk._response_headers = captured_headers
                            is_first_chunk = False
                        yield chunk

                    # Tool calls accumulation
                    if "tool_calls" in delta:
                        for tc in delta["tool_calls"]:
                            tc_index = tc.get("index", 0)
                            if tc_index not in current_tool_calls:
                                current_tool_calls[tc_index] = {
                                    "id": tc.get("id", ""),
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
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

                    # Finish
                    if finish_reason:
                        # Emit accumulated tool calls
                        if current_tool_calls and finish_reason == "tool_calls":
                            tool_calls_list = [
                                {"index": i, **current_tool_calls[i]}
                                for i in sorted(current_tool_calls.keys())
                            ]
                            chunk = litellm.ModelResponse(
                                id=response_id,
                                created=created,
                                model=f"gitlab_duo/{proxy_model}",
                                object="chat.completion.chunk",
                                choices=[
                                    {
                                        "index": index,
                                        "delta": {"tool_calls": tool_calls_list},
                                        "finish_reason": None,
                                    }
                                ],
                            )
                            if is_first_chunk:
                                chunk._response_headers = captured_headers
                                is_first_chunk = False
                            yield chunk

                        # Final chunk
                        final_chunk = litellm.ModelResponse(
                            id=response_id,
                            created=created,
                            model=f"gitlab_duo/{proxy_model}",
                            object="chat.completion.chunk",
                            choices=[
                                {
                                    "index": index,
                                    "delta": {},
                                    "finish_reason": finish_reason,
                                }
                            ],
                        )

                        usage = evt.get("usage")
                        if usage:
                            final_chunk.usage = litellm.Usage(
                                prompt_tokens=usage.get("prompt_tokens", 0),
                                completion_tokens=usage.get("completion_tokens", 0),
                                total_tokens=usage.get("total_tokens", 0),
                            )

                        if is_first_chunk:
                            final_chunk._response_headers = captured_headers
                            is_first_chunk = False
                        yield final_chunk

    # =========================================================================
    # ERROR PARSING
    # =========================================================================

    @classmethod
    def parse_quota_error(
        cls, error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Parse GitLab Duo quota/rate-limit errors.

        Distinguishes between:
        - 402 with credit exhaustion markers → CREDITS_EXHAUSTED (long cooldown, auto_remove)
        - 402 without markers → TRANSIENT_CREDIT_ERROR (short retry)
        - 429 → RATE_LIMITED
        - 403 with credit keywords → QUOTA_EXHAUSTED
        """
        if not isinstance(error, httpx.HTTPStatusError):
            return None

        status = error.response.status_code
        body = error_body or ""

        # Rate limited
        if status == 429:
            retry_after = 60  # Default
            ra = error.response.headers.get("retry-after")
            if ra:
                try:
                    retry_after = int(ra)
                except (ValueError, TypeError):
                    pass
            return {
                "retry_after": retry_after,
                "reason": "RATE_LIMITED",
                "reset_timestamp": None,
                "quota_reset_timestamp": None,
            }

        # 402 — distinguish real exhaustion from transient
        if status == 402:
            if cls._is_credit_exhaustion(body):
                return {
                    "retry_after": CREDIT_EXHAUSTION_COOLDOWN,
                    "reason": "CREDITS_EXHAUSTED",
                    "reset_timestamp": None,
                    "quota_reset_timestamp": None,
                }
            else:
                # Transient 402 — rotate to another credential
                # Use retry_after > 10 (small_cooldown_threshold) to ensure rotation
                # rather than retrying the same key, since internal retries already
                # happened in _stream_anthropic_with_retry
                return {
                    "retry_after": 30,
                    "reason": "TRANSIENT_CREDIT_ERROR",
                    "reset_timestamp": None,
                    "quota_reset_timestamp": None,
                }

        # Credit exhaustion via 403
        if status == 403:
            body_lower = body.lower()
            if any(kw in body_lower for kw in ("credit", "exhaust", "quota", "limit")):
                return {
                    "retry_after": 3600,  # 1 hour cooldown
                    "reason": "QUOTA_EXHAUSTED",
                    "reset_timestamp": None,
                    "quota_reset_timestamp": None,
                }

        return None

    # =========================================================================
    # CREDIT TRACKING
    # =========================================================================

    def record_credit_usage(self, api_key: str, model: str) -> None:
        """Record credit consumption for a successful request."""
        clean_model = model.split("/", 1)[1] if "/" in model else model
        cost = CREDIT_COSTS.get(clean_model, 0.5)  # Default 0.5 if unknown
        current = self._credit_usage.get(api_key, 0.0)
        self._credit_usage[api_key] = current + cost
        lib_logger.debug(
            "[GitLabDuo] Credit usage for ...%s: +%.3f (total: %.3f)",
            api_key[-8:] if len(api_key) > 8 else api_key[:4],
            cost,
            self._credit_usage[api_key],
        )

    def get_credit_limit(self, api_key: str) -> float:
        """Get the credit limit for a credential."""
        if api_key in self._credit_limits:
            return self._credit_limits[api_key]

        # Check per-credential env var (e.g., GITLAB_DUO_CREDITS_PER_ACCOUNT_1)
        # Try to find the credential index
        global_limit = float(
            os.getenv(
                "GITLAB_DUO_CREDITS_PER_ACCOUNT", str(DEFAULT_CREDITS_PER_ACCOUNT)
            )
        )
        self._credit_limits[api_key] = global_limit
        return global_limit

    def get_credits_info(self) -> Dict[str, Any]:
        """Get credit usage info for all tracked credentials.

        Returns a dict suitable for inclusion in quota stats.
        """
        total_limit = 0.0
        total_used = 0.0
        per_credential: Dict[str, Dict[str, Any]] = {}

        for api_key in list(self._credit_usage.keys()):
            limit = self.get_credit_limit(api_key)
            used = self._credit_usage.get(api_key, 0.0)
            remaining = max(0.0, limit - used)
            pct = round(remaining / limit * 100, 1) if limit > 0 else 0

            masked = api_key[-8:] if len(api_key) > 8 else api_key[:4]
            per_credential[masked] = {
                "limit": round(limit, 2),
                "used": round(used, 2),
                "remaining": round(remaining, 2),
                "pct_remaining": pct,
                "strikes": self._exhaustion_strikes.get(api_key, 0),
            }
            total_limit += limit
            total_used += used

        total_remaining = max(0.0, total_limit - total_used)

        return {
            "total_across_accounts": round(total_limit, 2),
            "used_across_accounts": round(total_used, 2),
            "remaining_across_accounts": round(total_remaining, 2),
            "per_credential": per_credential,
        }
