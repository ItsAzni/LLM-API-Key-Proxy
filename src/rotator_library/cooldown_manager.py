# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import asyncio
import logging
import time
from typing import Dict

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
        self._provider_locks: Dict[str, asyncio.Lock] = {}
        # Protects the _provider_locks dict itself (lazy init)
        self._locks_lock = asyncio.Lock()

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
        # Fast path: lock already exists
        if provider in self._provider_locks:
            return self._provider_locks[provider]
        # Slow path: create lock under protection
        async with self._locks_lock:
            if provider not in self._provider_locks:
                self._provider_locks[provider] = asyncio.Lock()
            return self._provider_locks[provider]

    async def is_cooling_down(self, credential: str) -> bool:
        """Checks if a credential is currently in a cooldown period."""
        provider = self._extract_provider(credential)
        lock = await self._get_provider_lock(provider)
        async with lock:
            return credential in self._cooldowns and time.time() < self._cooldowns[credential]

    async def start_cooldown(self, credential: str, duration: int):
        """
        Initiates or extends a cooldown period for a credential.
        The cooldown is set to the current time plus the specified duration.
        """
        provider = self._extract_provider(credential)
        lock = await self._get_provider_lock(provider)
        async with lock:
            self._cooldowns[credential] = time.time() + duration

    async def get_cooldown_remaining(self, credential: str) -> float:
        """
        Returns the remaining cooldown time in seconds for a credential.
        Returns 0 if the credential is not in a cooldown period.
        """
        provider = self._extract_provider(credential)
        lock = await self._get_provider_lock(provider)
        async with lock:
            if credential in self._cooldowns:
                remaining = self._cooldowns[credential] - time.time()
                return max(0, remaining)
            return 0
