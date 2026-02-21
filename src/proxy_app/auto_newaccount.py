# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel

"""
Auto-newaccount manager for GitLab Duo credentials.

When a GitLab Duo credential is removed from the pool (exhaustion via repeated
402 errors, or immediate 403 Forbidden), this module:
1. Sends a Telegram notification alerting the user
2. Queues a replacement account creation
3. Processes the queue strictly one-at-a-time with a 60 s cooldown between jobs
4. Reports success or failure on Telegram for each replacement

Enabled by default when TELEGRAM_BOT_TOKEN is set and gitlab_duo credentials
exist. Set GITLAB_DUO_AUTO_NEWACCOUNT=false to disable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from io import StringIO
from pathlib import Path
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)

# Seconds to wait between consecutive account creations so that browser
# sessions / temp-email inboxes don't collide.
_INTER_CREATION_COOLDOWN = 60


class AutoNewAccountManager:
    """Queue-based manager that creates replacement GitLab Duo credentials
    strictly one at a time."""

    def __init__(
        self,
        *,
        proxy_port: int = 8000,
        proxy_api_key: Optional[str] = None,
    ) -> None:
        # Telegram config
        self._telegram_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
        self._telegram_users: List[int] = self._parse_allowed_users()

        # Proxy config for reload calls
        self._proxy_port = proxy_port
        self._proxy_api_key = proxy_api_key or os.getenv("PROXY_API_KEY", "")

        # FIFO queue — each item is a credential path/string that was removed
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._max_retries = 2

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the background queue worker.  Call once at startup."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(
                self._process_queue(), name="auto-newaccount-worker"
            )
            logger.info("Auto-newaccount queue worker started")

    # ------------------------------------------------------------------
    # Public callback (registered on the provider)
    # ------------------------------------------------------------------

    async def on_credential_exhausted(self, credential: str) -> None:
        """Called by the provider when a credential is removed from the pool.

        Sends an immediate Telegram alert and enqueues a replacement job.
        """
        cred_name = (
            Path(credential).name
            if "/" in credential or "\\" in credential
            else credential
        )
        remaining = self._count_remaining_credentials(exclude=credential)

        queued = self._queue.qsize()
        queue_info = f"\n📋 Queue: {queued} pending" if queued > 0 else ""

        await self._send_telegram(
            f"⚠️ *GitLab Duo credential removed*\n\n"
            f"📁 `{cred_name}`\n"
            f"🗑️ Removed from pool ({remaining} remaining)\n\n"
            f"⏳ Replacement queued...{queue_info}"
        )

        await self._queue.put(credential)

    # ------------------------------------------------------------------
    # Background queue worker
    # ------------------------------------------------------------------

    async def _process_queue(self) -> None:
        """Process replacement requests strictly one at a time, forever."""
        while True:
            credential = await self._queue.get()
            try:
                await self._process_single_replacement(credential)

                # Cooldown before the next job so browser sessions / emails
                # don't collide.
                if not self._queue.empty():
                    pending = self._queue.qsize()
                    await self._send_telegram(
                        f"⏳ Waiting {_INTER_CREATION_COOLDOWN}s before next "
                        f"replacement ({pending} still queued)"
                    )
                    await asyncio.sleep(_INTER_CREATION_COOLDOWN)
            except Exception:
                logger.exception("auto-newaccount queue worker error")
            finally:
                self._queue.task_done()

    async def _process_single_replacement(self, credential: str) -> None:
        """Attempt to create one replacement credential (with retries)."""
        errors: list[str] = []
        for attempt in range(1, self._max_retries + 1):
            try:
                result = await self._create_and_reload()
                await self._send_telegram(
                    f"✅ *Replacement credential ready*\n\n"
                    f"📁 `{result['cred_name']}`\n"
                    f"📧 `{result['email']}`\n"
                    f"🔄 Proxy reloaded"
                )
                return
            except Exception as exc:
                msg = str(exc)
                if len(msg) > 300:
                    msg = msg[:300] + "..."
                errors.append(f"Attempt {attempt}: {msg}")
                logger.exception(
                    "auto-newaccount attempt %d/%d failed",
                    attempt,
                    self._max_retries,
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(5)

        # All attempts exhausted
        error_lines = "\n".join(errors)
        await self._send_telegram(
            f"❌ *Auto-newaccount failed*\n\n"
            f"{error_lines}\n\n"
            f"👉 Run `/newaccount` manually"
        )

    # ------------------------------------------------------------------
    # Core logic — create account + reload
    # ------------------------------------------------------------------

    async def _create_and_reload(self) -> dict:
        """Create a new GitLab trial account, perform OAuth, reload credentials.

        Returns dict with ``cred_name`` and ``email`` on success.
        Raises on failure.
        """
        # Late imports — these are heavy and optional
        from rotator_library.providers.gitlab_duo_provider import (
            DEFAULT_OAUTH_CALLBACK_PORT,
            DEFAULT_OAUTH_CLIENT_ID,
            GitLabDuoProvider,
            _get_instance_url,
        )
        from rotator_library.providers.utilities.gitlab_trial_automation import (
            GitLabTrialAutomator,
        )
        from rich.console import Console as RichConsole

        instance_url = _get_instance_url()
        client_id = os.getenv("GITLAB_OAUTH_CLIENT_ID", DEFAULT_OAUTH_CLIENT_ID)
        callback_port = int(
            os.getenv("GITLAB_DUO_OAUTH_PORT", str(DEFAULT_OAUTH_CALLBACK_PORT))
        )

        if not client_id:
            raise RuntimeError("GITLAB_OAUTH_CLIENT_ID is not set")

        # Determine output path (next available index)
        oauth_dir = Path("oauth_creds")
        oauth_dir.mkdir(exist_ok=True)
        idx = 1
        while (oauth_dir / f"gitlab_duo_oauth_{idx}.json").exists():
            idx += 1
        output_path = str(oauth_dir / f"gitlab_duo_oauth_{idx}.json")

        # Build a silent console for the automator
        _quiet_console = RichConsole(file=StringIO(), quiet=True)

        automator = GitLabTrialAutomator(
            console=_quiet_console,
            progress_callback=None,
        )

        async def oauth_runner(auth_url_handler):
            """Run the GitLab OAuth PKCE flow."""
            return await GitLabDuoProvider.oauth_setup(
                instance_url=instance_url,
                client_id=client_id,
                callback_port=callback_port,
                output_path=output_path,
                auth_url_handler=auth_url_handler,
                auto_open_browser=False,
            )

        # Run the full automation
        auto_result = await automator.run(oauth_runner)
        saved_path = auto_result.oauth_path

        # If no group was created the credential is useless
        if not auto_result.group_path:
            try:
                os.remove(saved_path)
            except OSError:
                pass
            raise RuntimeError(
                f"Account {auto_result.email} created but group activation failed"
            )

        # Write proxy metadata
        try:
            with open(saved_path, "r") as f:
                creds = json.load(f)
            metadata = creds.setdefault("_proxy_metadata", {})
            metadata["email"] = auto_result.email
            metadata["gitlab_duo_group"] = auto_result.group_path or ""
            metadata["gitlab_trial_automated"] = True
            metadata["created_via"] = "auto_newaccount"
            with open(saved_path, "w") as f:
                json.dump(creds, f, indent=2)
        except Exception as exc:
            logger.error("Failed to write proxy metadata to %s: %s", saved_path, exc)

        # Hot-reload credentials into the running proxy
        await self._reload_credentials()

        return {
            "cred_name": Path(saved_path).name,
            "email": auto_result.email,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _reload_credentials(self) -> bool:
        """POST to the proxy's own ``/v1/reload-credentials`` endpoint."""
        url = f"http://127.0.0.1:{self._proxy_port}/v1/reload-credentials"
        headers: dict[str, str] = {}
        if self._proxy_api_key:
            headers["Authorization"] = f"Bearer {self._proxy_api_key}"

        try:
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                resp = await http_client.post(url, headers=headers)
                if resp.status_code == 200:
                    logger.info("Credential reload succeeded: %s", resp.json())
                    return True
                else:
                    logger.error(
                        "Credential reload returned HTTP %d: %s",
                        resp.status_code,
                        resp.text[:200],
                    )
                    return False
        except Exception as exc:
            logger.error("Failed to call /v1/reload-credentials: %s", exc)
            return False

    async def _send_telegram(self, text: str) -> None:
        """Send a Telegram message via the Bot API (direct HTTP, no library needed)."""
        if not self._telegram_token or not self._telegram_users:
            logger.warning(
                "Telegram notification skipped (no token or no allowed users configured)"
            )
            return

        chat_id = self._telegram_users[0]
        url = f"https://api.telegram.org/bot{self._telegram_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as http_client:
                resp = await http_client.post(url, json=payload)
                if resp.status_code != 200:
                    logger.error(
                        "Telegram sendMessage failed (HTTP %d): %s",
                        resp.status_code,
                        resp.text[:200],
                    )
        except Exception as exc:
            logger.error("Telegram notification failed: %s", exc)

    def _count_remaining_credentials(self, exclude: str = "") -> int:
        """Count GitLab Duo OAuth credential files on disk, excluding *exclude*."""
        oauth_dir = Path("oauth_creds")
        if not oauth_dir.is_dir():
            return 0
        count = 0
        for f in oauth_dir.glob("gitlab_duo_oauth_*.json"):
            if str(f) != exclude and f.name != Path(exclude).name:
                count += 1
        return count

    @staticmethod
    def _parse_allowed_users() -> List[int]:
        """Parse ``TELEGRAM_ALLOWED_USERS`` env var into a list of integer user IDs."""
        raw = os.getenv("TELEGRAM_ALLOWED_USERS", "")
        users: List[int] = []
        for part in raw.split(","):
            part = part.strip()
            if part:
                try:
                    users.append(int(part))
                except ValueError:
                    pass
        return users
