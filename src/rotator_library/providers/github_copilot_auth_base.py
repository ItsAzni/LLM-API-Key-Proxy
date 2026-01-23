# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/providers/github_copilot_auth_base.py
"""
GitHub Copilot OAuth Base Class

Base class for GitHub Copilot OAuth authentication using GitHub Device Flow.
Handles device flow authentication, token storage, and API headers.

OAuth Configuration:
- Client ID: Ov23li8tweQw6odWQebz (GitHub Copilot official)
- Device Code URL: https://github.com/login/device/code
- Access Token URL: https://github.com/login/oauth/access_token
- Scope: read:user
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markup import escape as rich_escape

from ..utils.headless_detection import is_headless_environment
from ..utils.resilient_io import safe_write_json

lib_logger = logging.getLogger("rotator_library")
console = Console()

# =============================================================================
# OAUTH CONFIGURATION
# =============================================================================

# GitHub Copilot OAuth client ID (official from opencode reference)
CLIENT_ID = "Ov23li8tweQw6odWQebz"

# Default GitHub.com OAuth endpoints
DEFAULT_DEVICE_CODE_URL = "https://github.com/login/device/code"
DEFAULT_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"

# OAuth scope
OAUTH_SCOPE = "read:user"

# Polling safety margin to avoid clock skew issues (3 seconds)
OAUTH_POLLING_SAFETY_MARGIN_SECONDS = 3

# Token never expires for GitHub OAuth (long-lived personal access token)
# We store 0 to indicate no expiry
TOKEN_NEVER_EXPIRES = 0


def _normalize_domain(url: str) -> str:
    """Normalize a GitHub domain URL by stripping protocol and trailing slash."""
    return url.replace("https://", "").replace("http://", "").rstrip("/")


def _get_urls(domain: str) -> Dict[str, str]:
    """Get OAuth endpoints for a given GitHub domain."""
    return {
        "device_code_url": f"https://{domain}/login/device/code",
        "access_token_url": f"https://{domain}/login/oauth/access_token",
    }


@dataclass
class CredentialSetupResult:
    """
    Standardized result structure for credential setup operations.
    """
    success: bool
    file_path: Optional[str] = None
    email: Optional[str] = None
    tier: Optional[str] = None
    is_update: bool = False
    error: Optional[str] = None
    credentials: Optional[Dict[str, Any]] = field(default=None, repr=False)


class GitHubCopilotAuthBase:
    """
    Base class for GitHub Copilot OAuth authentication.

    Uses GitHub Device Flow OAuth for authentication, which is ideal for
    CLI tools as it doesn't require a local callback server.

    Subclasses may override:
        - ENV_PREFIX: Prefix for environment variables (default: "GITHUB_COPILOT")
    """

    # Configuration
    CLIENT_ID: str = CLIENT_ID
    ENV_PREFIX: str = "GITHUB_COPILOT"
    USER_AGENT: str = "LLM-API-Key-Proxy/1.0"

    def __init__(self):
        self._credentials_cache: Dict[str, Dict[str, Any]] = {}
        self._locks_lock = asyncio.Lock()
        self._refresh_locks: Dict[str, asyncio.Lock] = {}

        # Tracking for unavailable credentials
        self._unavailable_credentials: Dict[str, float] = {}
        self._unavailable_ttl_seconds: int = 360

    # =========================================================================
    # LOCK MANAGEMENT
    # =========================================================================

    async def _get_lock(self, path: str) -> asyncio.Lock:
        """Get or create a lock for a credential path."""
        async with self._locks_lock:
            if path not in self._refresh_locks:
                self._refresh_locks[path] = asyncio.Lock()
            return self._refresh_locks[path]

    # =========================================================================
    # ENVIRONMENT VARIABLE LOADING
    # =========================================================================

    def _parse_env_credential_path(self, path: str) -> Optional[str]:
        """Parse a virtual env:// path and return the credential index."""
        if not path.startswith("env://"):
            return None
        parts = path[6:].split("/")
        if len(parts) >= 2:
            return parts[1]
        return "0"

    def _load_from_env(self, credential_index: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load OAuth credentials from environment variables.

        Expected variables:
        - {ENV_PREFIX}_{N}_ACCESS_TOKEN or {ENV_PREFIX}_ACCESS_TOKEN
        - {ENV_PREFIX}_{N}_ENTERPRISE_DOMAIN (optional)
        """
        if credential_index and credential_index != "0":
            prefix = f"{self.ENV_PREFIX}_{credential_index}"
            default_id = f"env-user-{credential_index}"
        else:
            prefix = self.ENV_PREFIX
            default_id = "env-user"

        access_token = os.getenv(f"{prefix}_ACCESS_TOKEN")
        if not access_token:
            return None

        lib_logger.debug(f"Loading {prefix} credentials from environment variables")

        enterprise_domain = os.getenv(f"{prefix}_ENTERPRISE_DOMAIN")

        creds = {
            "access_token": access_token,
            "expiry_date": TOKEN_NEVER_EXPIRES,
            "_proxy_metadata": {
                "user_id": default_id,
                "last_check_timestamp": time.time(),
                "loaded_from_env": True,
                "env_credential_index": credential_index or "0",
            },
        }

        if enterprise_domain:
            creds["_proxy_metadata"]["enterprise_domain"] = _normalize_domain(enterprise_domain)

        return creds

    # =========================================================================
    # CREDENTIAL LOADING AND SAVING
    # =========================================================================

    async def _load_credentials(self, path: str) -> Dict[str, Any]:
        """Load credentials from file or environment."""
        if path in self._credentials_cache:
            return self._credentials_cache[path]

        async with await self._get_lock(path):
            if path in self._credentials_cache:
                return self._credentials_cache[path]

            # Check if this is a virtual env:// path
            credential_index = self._parse_env_credential_path(path)
            if credential_index is not None:
                env_creds = self._load_from_env(credential_index)
                if env_creds:
                    self._credentials_cache[path] = env_creds
                    return env_creds
                else:
                    raise IOError(
                        f"Environment variables for {self.ENV_PREFIX} credential index {credential_index} not found"
                    )

            # Try file-based loading
            try:
                lib_logger.debug(f"Loading {self.ENV_PREFIX} credentials from file: {path}")
                with open(path, "r") as f:
                    creds = json.load(f)
                self._credentials_cache[path] = creds
                return creds
            except FileNotFoundError:
                env_creds = self._load_from_env()
                if env_creds:
                    lib_logger.info(
                        f"File '{path}' not found, using {self.ENV_PREFIX} credentials from environment variables"
                    )
                    self._credentials_cache[path] = env_creds
                    return env_creds
                raise IOError(
                    f"{self.ENV_PREFIX} OAuth credential file not found at '{path}'"
                )
            except Exception as e:
                raise IOError(
                    f"Failed to load {self.ENV_PREFIX} OAuth credentials from '{path}': {e}"
                )

    async def _save_credentials(self, path: str, creds: Dict[str, Any]):
        """Save credentials with in-memory fallback if disk unavailable."""
        self._credentials_cache[path] = creds

        if creds.get("_proxy_metadata", {}).get("loaded_from_env"):
            lib_logger.debug("Credentials loaded from env, skipping file save")
            return

        if safe_write_json(
            path, creds, lib_logger, secure_permissions=True, buffer_on_failure=True
        ):
            lib_logger.debug(f"Saved updated {self.ENV_PREFIX} OAuth credentials to '{path}'.")
        else:
            lib_logger.warning(
                f"Credentials for {self.ENV_PREFIX} cached in memory only (buffered for retry)."
            )

    # =========================================================================
    # TOKEN INITIALIZATION
    # =========================================================================

    async def initialize_token(
        self,
        creds_or_path: str | Dict[str, Any],
        force_interactive: bool = False,
    ) -> Dict[str, Any]:
        """
        Initialize and validate OAuth token.

        For GitHub Copilot, OAuth tokens are long-lived and don't require refresh.
        This method validates the token exists and returns the credentials.

        Args:
            creds_or_path: Either a credentials dict or path to credentials file.
            force_interactive: If True, forces re-authentication via device flow.

        Returns:
            Validated credentials dict.

        Raises:
            IOError: If credentials cannot be loaded or are invalid.
        """
        # Determine if input is a path or credentials dict
        if isinstance(creds_or_path, str):
            path = creds_or_path
            display_name = Path(path).name
            creds = await self._load_credentials(path)
        else:
            path = None
            creds = creds_or_path
            display_name = creds.get("_proxy_metadata", {}).get(
                "display_name", "in-memory object"
            )

        lib_logger.debug(
            f"Initializing {self.ENV_PREFIX} token for '{display_name}'..."
        )

        # Validate access token exists
        access_token = creds.get("access_token")
        if not access_token:
            raise IOError(
                f"{self.ENV_PREFIX} credentials at '{display_name}' missing access_token"
            )

        # GitHub OAuth tokens are long-lived, no refresh needed
        # force_interactive is accepted for API compatibility but not used
        _ = force_interactive  # Silence unused variable warning

        lib_logger.debug(
            f"{self.ENV_PREFIX} token for '{display_name}' is valid."
        )

        return creds

    # =========================================================================
    # DEVICE FLOW OAUTH
    # =========================================================================

    async def _perform_device_flow_oauth(
        self,
        domain: str = "github.com",
        display_name: str = "GitHub Copilot"
    ) -> Dict[str, Any]:
        """
        Perform GitHub Device Flow OAuth authentication.

        This flow is ideal for CLI tools as it:
        1. Requests a device code from GitHub
        2. Shows user a URL and code to enter
        3. Polls for authorization completion
        4. Returns access token on success

        Args:
            domain: GitHub domain (github.com or enterprise domain)
            display_name: Display name for the credential being set up

        Returns:
            Credentials dict with access_token and metadata
        """
        is_headless = is_headless_environment()
        urls = _get_urls(domain)

        # Step 1: Request device code
        async with httpx.AsyncClient() as client:
            device_response = await client.post(
                urls["device_code_url"],
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "User-Agent": self.USER_AGENT,
                },
                json={
                    "client_id": self.CLIENT_ID,
                    "scope": OAUTH_SCOPE,
                },
            )

            if not device_response.is_success:
                raise Exception(
                    f"Failed to initiate device authorization: {device_response.status_code} {device_response.text}"
                )

            device_data = device_response.json()
            verification_uri = device_data.get("verification_uri")
            user_code = device_data.get("user_code")
            device_code = device_data.get("device_code")
            interval = device_data.get("interval", 5)

            if not all([verification_uri, user_code, device_code]):
                raise Exception(f"Invalid device code response: {device_data}")

        # Step 2: Show user the verification URL and code
        if is_headless:
            auth_panel_text = Text.from_markup(
                "Running in headless environment (no GUI detected).\n"
                "Please open the URL below in a browser on another machine:\n"
            )
        else:
            auth_panel_text = Text.from_markup(
                "1. Open the URL below in your browser.\n"
                "2. Enter the code when prompted.\n"
            )

        console.print(
            Panel(
                auth_panel_text,
                title=f"GitHub Copilot OAuth Setup for [bold yellow]{display_name}[/bold yellow]",
                style="bold blue",
            )
        )

        escaped_url = rich_escape(verification_uri)
        console.print(f"\n[bold]URL:[/bold] [link={verification_uri}]{escaped_url}[/link]")
        console.print(f"[bold]Code:[/bold] [bold green]{user_code}[/bold green]\n")

        # Open browser if not headless
        if not is_headless:
            try:
                import webbrowser
                webbrowser.open(verification_uri)
                lib_logger.info("Browser opened successfully for OAuth flow")
            except Exception as e:
                lib_logger.warning(
                    f"Failed to open browser automatically: {e}. Please open the URL manually."
                )

        # Step 3: Poll for authorization
        async with httpx.AsyncClient() as client:
            with console.status(
                "[bold green]Waiting for you to complete authentication in the browser...[/bold green]",
                spinner="dots",
            ):
                while True:
                    await asyncio.sleep(interval + OAUTH_POLLING_SAFETY_MARGIN_SECONDS)

                    token_response = await client.post(
                        urls["access_token_url"],
                        headers={
                            "Accept": "application/json",
                            "Content-Type": "application/json",
                            "User-Agent": self.USER_AGENT,
                        },
                        json={
                            "client_id": self.CLIENT_ID,
                            "device_code": device_code,
                            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                        },
                    )

                    if not token_response.is_success:
                        raise Exception(f"Token request failed: {token_response.text}")

                    token_data = token_response.json()

                    # Check for access token
                    if "access_token" in token_data:
                        access_token = token_data["access_token"]

                        creds = {
                            "access_token": access_token,
                            "expiry_date": TOKEN_NEVER_EXPIRES,
                            "_proxy_metadata": {
                                "last_check_timestamp": time.time(),
                                "domain": domain,
                            },
                        }

                        # Mark as enterprise if not github.com
                        if domain != "github.com":
                            creds["_proxy_metadata"]["enterprise_domain"] = domain

                        lib_logger.info(
                            f"GitHub Copilot OAuth completed successfully for '{display_name}'"
                        )
                        return creds

                    # Handle polling states
                    error = token_data.get("error")

                    if error == "authorization_pending":
                        # User hasn't authorized yet, continue polling
                        continue

                    elif error == "slow_down":
                        # Need to slow down polling
                        new_interval = token_data.get("interval", interval + 5)
                        interval = new_interval
                        continue

                    elif error == "expired_token":
                        raise Exception("Device code expired. Please try again.")

                    elif error == "access_denied":
                        raise Exception("Authorization was denied by the user.")

                    elif error:
                        raise Exception(f"OAuth error: {error}")

                    # Unknown state, continue polling
                    continue

    # =========================================================================
    # AUTH HEADER
    # =========================================================================

    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """
        Get authorization header for GitHub Copilot API requests.

        Args:
            credential_identifier: Path to credential file or env:// path

        Returns:
            Dict with Authorization header
        """
        try:
            creds = await self._load_credentials(credential_identifier)
            access_token = creds.get("access_token")

            if not access_token:
                raise ValueError(f"No access_token found in credentials at '{credential_identifier}'")

            return {"Authorization": f"Bearer {access_token}"}

        except Exception as e:
            # Try cached credentials as fallback
            cached = self._credentials_cache.get(credential_identifier)
            if cached and cached.get("access_token"):
                lib_logger.warning(
                    f"Credential load failed for {credential_identifier}: {e}. Using cached token."
                )
                return {"Authorization": f"Bearer {cached['access_token']}"}
            raise

    def is_credential_available(self, path: str) -> bool:
        """Check if a credential is available for rotation."""
        if path in self._unavailable_credentials:
            marked_time = self._unavailable_credentials.get(path)
            if marked_time is not None:
                now = time.time()
                if now - marked_time > self._unavailable_ttl_seconds:
                    self._unavailable_credentials.pop(path, None)
                else:
                    return False
        return True

    # =========================================================================
    # CREDENTIAL MANAGEMENT
    # =========================================================================

    def _get_provider_file_prefix(self) -> str:
        """Get the file name prefix for this provider's credential files."""
        return "github_copilot"

    def _get_oauth_base_dir(self) -> Path:
        """Get the base directory for OAuth credential files."""
        return Path.cwd() / "oauth_creds"

    def _find_existing_credential_by_token(
        self, access_token: str, base_dir: Optional[Path] = None
    ) -> Optional[Path]:
        """Find an existing credential file with the same access token."""
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        prefix = self._get_provider_file_prefix()
        pattern = str(base_dir / f"{prefix}_oauth_*.json")

        for cred_file in glob(pattern):
            try:
                with open(cred_file, "r") as f:
                    creds = json.load(f)
                existing_token = creds.get("access_token")
                if existing_token == access_token:
                    return Path(cred_file)
            except Exception:
                continue

        return None

    def _get_next_credential_number(self, base_dir: Optional[Path] = None) -> int:
        """Get the next available credential number."""
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        prefix = self._get_provider_file_prefix()
        pattern = str(base_dir / f"{prefix}_oauth_*.json")

        existing_numbers = []
        for cred_file in glob(pattern):
            match = re.search(r"_oauth_(\d+)\.json$", cred_file)
            if match:
                existing_numbers.append(int(match.group(1)))

        if not existing_numbers:
            return 1
        return max(existing_numbers) + 1

    def _build_credential_path(
        self, base_dir: Optional[Path] = None, number: Optional[int] = None
    ) -> Path:
        """Build a path for a new credential file."""
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        if number is None:
            number = self._get_next_credential_number(base_dir)

        prefix = self._get_provider_file_prefix()
        filename = f"{prefix}_oauth_{number}.json"
        return base_dir / filename

    async def setup_credential(
        self,
        base_dir: Optional[Path] = None,
        domain: str = "github.com"
    ) -> CredentialSetupResult:
        """
        Complete credential setup flow: Device Flow OAuth -> save -> discovery.

        Args:
            base_dir: Base directory for credential files
            domain: GitHub domain (github.com or enterprise domain)

        Returns:
            CredentialSetupResult with success status and file path
        """
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        base_dir.mkdir(exist_ok=True)

        try:
            display_name = f"new {self.ENV_PREFIX} credential"
            new_creds = await self._perform_device_flow_oauth(
                domain=domain,
                display_name=display_name
            )

            access_token = new_creds.get("access_token")
            if not access_token:
                return CredentialSetupResult(
                    success=False,
                    error="Could not retrieve access token from OAuth response"
                )

            # Check for existing credential with same token
            existing_path = self._find_existing_credential_by_token(access_token, base_dir)
            is_update = existing_path is not None

            if is_update:
                file_path = existing_path
            else:
                file_path = self._build_credential_path(base_dir)

            await self._save_credentials(str(file_path), new_creds)

            return CredentialSetupResult(
                success=True,
                file_path=str(file_path),
                is_update=is_update,
                credentials=new_creds,
            )

        except Exception as e:
            lib_logger.error(f"Credential setup failed: {e}")
            return CredentialSetupResult(success=False, error=str(e))

    def list_credentials(self, base_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        List all credential files for this provider.

        Args:
            base_dir: Base directory for credential files

        Returns:
            List of credential info dicts with file_path, number, and metadata
        """
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        prefix = self._get_provider_file_prefix()
        pattern = str(base_dir / f"{prefix}_oauth_*.json")

        credentials = []
        for cred_file in sorted(glob(pattern)):
            try:
                with open(cred_file, "r") as f:
                    creds = json.load(f)

                metadata = creds.get("_proxy_metadata", {})

                match = re.search(r"_oauth_(\d+)\.json$", cred_file)
                number = int(match.group(1)) if match else 0

                cred_info = {
                    "file_path": cred_file,
                    "number": number,
                    "domain": metadata.get("domain", "github.com"),
                }

                # Add enterprise domain if present
                if metadata.get("enterprise_domain"):
                    cred_info["enterprise_domain"] = metadata["enterprise_domain"]

                credentials.append(cred_info)
            except Exception:
                continue

        return credentials
