# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
GitLab Duo OAuth Auth Base Class

Handles OAuth 2.0 (PKCE) authentication for GitLab Duo provider.
Used by the credential_tool for interactive credential setup.

OAuth Flow:
  1. Generate PKCE code_verifier + S256 code_challenge
  2. Open browser to {instanceUrl}/oauth/authorize
  3. Start local callback server to receive authorization code
  4. Exchange code for tokens at {instanceUrl}/oauth/token
  5. Save credentials to oauth_creds/gitlab_duo_oauth_N.json

No client_secret is needed (public client with PKCE).
"""

from __future__ import annotations

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

from ..utils.headless_detection import is_headless_environment
from ..utils.resilient_io import safe_write_json

lib_logger = logging.getLogger("rotator_library")
console = Console()

# Import OAuth constants and helpers from the provider
from .gitlab_duo_provider import (
    DEFAULT_OAUTH_CLIENT_ID,
    DEFAULT_OAUTH_CALLBACK_PORT,
    OAUTH_REFRESH_BUFFER,
    _get_instance_url,
    GitLabDuoProvider,
)


@dataclass
class CredentialSetupResult:
    """Standardized result structure for credential setup operations."""

    success: bool
    file_path: Optional[str] = None
    email: Optional[str] = None
    tier: Optional[str] = None
    is_update: bool = False
    error: Optional[str] = None
    credentials: Optional[Dict[str, Any]] = field(default=None, repr=False)


class GitLabDuoAuthBase:
    """
    Auth base class for GitLab Duo OAuth credential management.

    Provides interactive OAuth 2.0 PKCE setup, credential listing,
    token validation, and env export — integrated with the credential_tool.
    """

    ENV_PREFIX: str = "GITLAB_DUO"

    def __init__(self):
        self._credentials_cache: Dict[str, Dict[str, Any]] = {}

    # =========================================================================
    # PROVIDER FILE PREFIX
    # =========================================================================

    def _get_provider_file_prefix(self) -> str:
        return "gitlab_duo"

    def _get_oauth_base_dir(self) -> Path:
        return Path.cwd() / "oauth_creds"

    # =========================================================================
    # CREDENTIAL LOADING
    # =========================================================================

    async def _load_credentials(self, path: str) -> Dict[str, Any]:
        """Load credentials from file."""
        if path in self._credentials_cache:
            return self._credentials_cache[path]

        try:
            with open(path, "r") as f:
                creds = json.load(f)
            self._credentials_cache[path] = creds
            return creds
        except FileNotFoundError:
            raise IOError(f"GitLab Duo OAuth credential file not found: {path}")
        except Exception as e:
            raise IOError(f"Failed to load GitLab Duo OAuth credentials: {e}")

    async def _save_credentials(self, path: str, creds: Dict[str, Any]):
        """Save credentials to file."""
        self._credentials_cache[path] = creds
        safe_write_json(path, creds, lib_logger, secure_permissions=True)

    # =========================================================================
    # TOKEN INITIALIZATION
    # =========================================================================

    async def initialize_token(
        self,
        creds_or_path: str | Dict[str, Any],
        force_interactive: bool = False,  # noqa: ARG002
    ) -> Dict[str, Any]:
        """
        Initialize and validate an OAuth token.

        Checks that access_token and refresh_token exist.
        If the token is expired, refreshes it automatically.
        """
        if isinstance(creds_or_path, str):
            path = creds_or_path
            creds = await self._load_credentials(path)
        else:
            path = None
            creds = creds_or_path

        access_token = creds.get("access_token")
        refresh_token = creds.get("refresh_token")

        if not access_token:
            raise IOError("GitLab Duo OAuth credentials missing access_token")
        if not refresh_token:
            raise IOError("GitLab Duo OAuth credentials missing refresh_token")

        # Check if token needs refresh
        expires_at = creds.get("expires_at", 0)
        if time.time() >= (expires_at - OAUTH_REFRESH_BUFFER):
            lib_logger.info("[GitLabDuo] Token expired during init, refreshing...")
            provider = GitLabDuoProvider()
            if path:
                creds = await provider._refresh_oauth_token(path, creds)
            else:
                lib_logger.warning(
                    "[GitLabDuo] Cannot refresh token without file path"
                )

        return creds

    # =========================================================================
    # INTERACTIVE OAUTH SETUP
    # =========================================================================

    async def setup_credential(
        self, base_dir: Optional[Path] = None
    ) -> CredentialSetupResult:
        """
        Complete credential setup flow: OAuth PKCE -> save -> ready.

        Opens browser for GitLab authentication, exchanges code for tokens,
        and saves to oauth_creds/gitlab_duo_oauth_N.json.
        """
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()
        base_dir.mkdir(exist_ok=True)

        is_headless = is_headless_environment()

        # Get configuration
        instance_url = _get_instance_url()
        client_id = os.getenv("GITLAB_OAUTH_CLIENT_ID", DEFAULT_OAUTH_CLIENT_ID)
        callback_port = int(
            os.getenv("GITLAB_DUO_OAUTH_PORT", str(DEFAULT_OAUTH_CALLBACK_PORT))
        )

        # Check client_id — only missing for self-hosted instances without config
        if not client_id:
            redirect_uri = f"http://127.0.0.1:{callback_port}/callback"
            console.print(
                Panel(
                    Text.from_markup(
                        "[bold red]GITLAB_OAUTH_CLIENT_ID is not set.[/bold red]\n\n"
                        "For self-managed GitLab instances, you must create an OAuth app:\n\n"
                        f"  1. Go to [bold cyan]{instance_url}/-/profile/applications[/bold cyan]\n"
                        "  2. Create a new application with:\n"
                        f"     [bold]Redirect URI:[/bold] {redirect_uri}\n"
                        "     [bold]Scopes:[/bold] api\n"
                        "     [bold]Confidential:[/bold] No (uncheck)\n"
                        "  3. Copy the [bold yellow]Application ID[/bold yellow] and add to .env:\n"
                        '     GITLAB_OAUTH_CLIENT_ID="<your_application_id>"'
                    ),
                    title="GitLab Duo OAuth - Setup Required",
                    style="bold yellow",
                )
            )
            return CredentialSetupResult(
                success=False,
                error=(
                    "GITLAB_OAUTH_CLIENT_ID not set. "
                    f"Create an OAuth app at {instance_url}/-/profile/applications "
                    f"with redirect URI: {redirect_uri}"
                ),
            )

        if is_headless:
            console.print(
                Panel(
                    Text.from_markup(
                        "[bold red]Headless environment detected.[/bold red]\n\n"
                        "GitLab Duo OAuth requires a browser for authentication.\n"
                        "Please run this on a machine with a browser, then copy\n"
                        "the resulting JSON file to oauth_creds/ on this server."
                    ),
                    title="GitLab Duo OAuth",
                    style="bold red",
                )
            )
            return CredentialSetupResult(
                success=False,
                error="Headless environment - browser required for GitLab OAuth",
            )

        console.print(
            Panel(
                Text.from_markup(
                    f"[bold]GitLab Instance:[/bold] {instance_url}\n"
                    f"[bold]Auth Method:[/bold] OAuth 2.0 with PKCE\n"
                    f"[bold]Callback Port:[/bold] {callback_port}\n\n"
                    "Your browser will open for GitLab authentication.\n"
                    "No client secret is needed (public client with PKCE)."
                ),
                title="GitLab Duo OAuth Setup",
                style="bold blue",
            )
        )

        try:
            # Determine output path
            next_num = self._get_next_credential_number(base_dir)
            output_path = str(
                base_dir / f"gitlab_duo_oauth_{next_num}.json"
            )

            # Run the OAuth flow (reuse the provider's oauth_setup)
            saved_path = await GitLabDuoProvider.oauth_setup(
                instance_url=instance_url,
                client_id=client_id,
                callback_port=callback_port,
                output_path=output_path,
            )

            # Check for existing credential with same token
            try:
                with open(saved_path, "r") as f:
                    new_creds = json.load(f)
                access_token = new_creds.get("access_token", "")
                existing = self._find_existing_credential_by_token(
                    access_token, base_dir, exclude_path=saved_path
                )
                if existing:
                    # Update existing instead of creating duplicate
                    import shutil
                    shutil.copy2(saved_path, str(existing))
                    os.remove(saved_path)
                    saved_path = str(existing)
                    return CredentialSetupResult(
                        success=True,
                        file_path=saved_path,
                        is_update=True,
                        credentials=new_creds,
                    )
            except Exception:
                pass

            # Try to get user info for display
            email = None
            try:
                with open(saved_path, "r") as f:
                    creds = json.load(f)
                email = await self._fetch_gitlab_user_email(creds)
                if email:
                    creds.setdefault("_proxy_metadata", {})["email"] = email
                    await self._save_credentials(saved_path, creds)
            except Exception as e:
                lib_logger.debug(f"Could not fetch GitLab user info: {e}")

            return CredentialSetupResult(
                success=True,
                file_path=saved_path,
                email=email,
                is_update=False,
            )

        except Exception as e:
            lib_logger.error(f"GitLab Duo OAuth setup failed: {e}")
            return CredentialSetupResult(success=False, error=str(e))

    async def _fetch_gitlab_user_email(self, creds: dict) -> Optional[str]:
        """Fetch the authenticated user's email from GitLab API."""
        instance_url = creds.get("instance_url", _get_instance_url())
        access_token = creds.get("access_token")
        if not access_token:
            return None

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{instance_url}/api/v4/user",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10.0,
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("email") or data.get("username")
        return None

    # =========================================================================
    # CREDENTIAL DISCOVERY
    # =========================================================================

    def _find_existing_credential_by_token(
        self,
        access_token: str,
        base_dir: Optional[Path] = None,
        exclude_path: Optional[str] = None,
    ) -> Optional[Path]:
        """Find an existing credential file with the same access token."""
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        prefix = self._get_provider_file_prefix()
        pattern = str(base_dir / f"{prefix}_oauth_*.json")

        for cred_file in glob(pattern):
            if exclude_path and os.path.abspath(cred_file) == os.path.abspath(
                exclude_path
            ):
                continue
            try:
                with open(cred_file, "r") as f:
                    creds = json.load(f)
                if creds.get("access_token") == access_token:
                    return Path(cred_file)
            except Exception:
                continue
        return None

    def _get_next_credential_number(
        self, base_dir: Optional[Path] = None
    ) -> int:
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

    def list_credentials(
        self, base_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        List all GitLab Duo OAuth credential files.

        Returns list of dicts with file_path, number, email, instance_url.
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
                    "email": metadata.get("email", "unknown"),
                    "instance_url": creds.get("instance_url", "https://gitlab.com"),
                }

                # Check token expiry
                expires_at = creds.get("expires_at", 0)
                if expires_at:
                    cred_info["expires_at"] = expires_at
                    cred_info["expired"] = time.time() >= expires_at

                credentials.append(cred_info)
            except Exception:
                continue

        return credentials

    # =========================================================================
    # ENV EXPORT
    # =========================================================================

    def build_env_lines(
        self, creds: Dict[str, Any], cred_number: int
    ) -> List[str]:
        """
        Generate .env file lines for a GitLab Duo OAuth credential.

        For stateless deployments (Docker, etc.) that use env vars.
        """
        metadata = creds.get("_proxy_metadata", {})
        email = metadata.get("email", "unknown")
        instance_url = creds.get("instance_url", "https://gitlab.com")
        prefix = f"{self.ENV_PREFIX}_{cred_number}"

        lines = [
            f"# GitLab Duo Credential #{cred_number} for: {email}",
            f"# Instance: {instance_url}",
            f"# Exported from: gitlab_duo_oauth_{cred_number}.json",
            f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"{prefix}_ACCESS_TOKEN=\"{creds.get('access_token', '')}\"",
            f"{prefix}_REFRESH_TOKEN=\"{creds.get('refresh_token', '')}\"",
        ]

        return lines
