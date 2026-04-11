# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import asyncio
import logging
import time
from typing import Dict

from .utils.provider_locks import ProviderLockManager

lib_logger = logging.getLogger("rotator_library")


class CooldownManager:
    """
    Manages cooldown periods for API credentials to handle rate limiting.
    Cooldowns are applied per-credential, allowing other credentials from the
    same provider to be used while one is cooling down.

    Uses per-provider sharded locks to avoid global serialization:
    parallel requests to different providers do not block each other.
    """

    def __init__(self):
        self._cooldowns: Dict[str, float] = {}
        # Per-provider locks: each provider has its own asyncio.Lock
        self._provider_lock_manager = ProviderLockManager()

    def _extract_provider(self, credential: str) -> str:
        """
        Extract provider name from a credential string.
        Credentials typically follow the pattern 'provider_key_N' or
        are just the credential string itself.
        Returns a provider key suitable for lock sharding.
        """
        # Use the first segment before '_' as provider identifier,
        # falling back to the full credential if no '_' is found.
        parts = credential.split("_")
        if len(parts) >= 2:
            return parts[0]
        return credential

    async def _get_provider_lock(self, provider: str) -> asyncio.Lock:
        """
        Lazily create and return the lock for a given provider.
        Uses a meta-lock to safely initialize new per-provider locks.
        """
        return await self._provider_lock_manager.get_lock(provider)

    async def is_cooling_down(self, credential: str) -> bool:
        """Checks if a credential is currently in a cooldown period."""
        # CPython dict reads are GIL-protected; no lock needed for a single lookup.
        expiry = self._cooldowns.get(credential)
        return expiry is not None and time.time() < expiry

    async def start_cooldown(self, credential: str, duration: int):
        """
        Initiates or extends a cooldown period for a credential.
        Sets expiry to max(existing, now + duration) so concurrent 429s
        with different durations always keep the longest cooldown.
        """
        provider = self._extract_provider(credential)
        lock = await self._get_provider_lock(provider)
        async with lock:
            new_expiry = time.time() + duration
            existing = self._cooldowns.get(credential, 0)
            self._cooldowns[credential] = max(existing, new_expiry)

    async def get_cooldown_remaining(self, credential: str) -> float:
        """
        Returns the remaining cooldown time in seconds for a credential.
        Returns 0 if the credential is not in a cooldown period.
        """
        # Single dict read — no lock needed in CPython asyncio context.
        expiry = self._cooldowns.get(credential)
        if expiry is None:
            return 0
        return max(0.0, expiry - time.time())
