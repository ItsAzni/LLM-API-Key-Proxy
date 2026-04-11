# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import asyncio
from typing import Dict


class ProviderLockManager:
    """Lazily creates and returns per-provider asyncio locks."""

    def __init__(self):
        self._provider_locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()

    async def get_lock(self, provider: str) -> asyncio.Lock:
        """Get or create the lock for a provider."""
        if provider in self._provider_locks:
            return self._provider_locks[provider]

        async with self._locks_lock:
            if provider not in self._provider_locks:
                self._provider_locks[provider] = asyncio.Lock()
            return self._provider_locks[provider]
