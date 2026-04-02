# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/providers/auth_queue_mixin.py

import time
import asyncio
import logging
from pathlib import Path

lib_logger = logging.getLogger("rotator_library")


class AuthQueueMixin:
    """Shared mixin providing deduplicated refresh/re-auth queue processing methods.

    Used by GoogleOAuthBase, IFlowAuthBase, and QwenAuthBase.
    Requires the consuming class to provide these attributes/methods:
        - _queue_retry_count: dict tracking retry counts per credential path
        - _refresh_max_retries: int, maximum retry attempts
        - _queued_credentials: set of currently queued credential paths
        - _refresh_queue: asyncio.Queue for normal refresh operations
        - _reauth_queue: asyncio.Queue for re-auth operations
        - _queue_tracking_lock: asyncio.Lock for queue tracking
        - _reauth_processor_task: task reference for the re-auth processor
        - _queue_processor_task: task reference for the refresh processor
        - _unavailable_credentials: dict tracking unavailable credentials
        - initialize_token(path, force_interactive=True): async method
    """

    async def _handle_refresh_failure(self, path: str, force: bool, error: str):
        """Handle a refresh failure with back-of-line retry logic.

        - Increments retry count
        - If under max retries: re-adds to END of queue
        - If at max retries: kicks credential out (retried next BackgroundRefresher cycle)
        """
        retry_count = self._queue_retry_count.get(path, 0) + 1
        self._queue_retry_count[path] = retry_count

        if retry_count >= self._refresh_max_retries:
            # Kicked out until next BackgroundRefresher cycle
            lib_logger.error(
                f"Max retries ({self._refresh_max_retries}) reached for '{Path(path).name}' "
                f"(last error: {error}). Will retry next refresh cycle."
            )
            self._queue_retry_count.pop(path, None)
            async with self._queue_tracking_lock:
                self._queued_credentials.discard(path)
            return

        # Re-add to END of queue for retry
        lib_logger.warning(
            f"Refresh failed for '{Path(path).name}' ({error}). "
            f"Retry {retry_count}/{self._refresh_max_retries}, back of queue."
        )
        # Keep in queued_credentials set, add back to queue
        await self._refresh_queue.put((path, force))

    async def _process_reauth_queue(self):
        """Background worker that processes re-auth requests.

        Key behaviors:
        - Credentials ARE marked unavailable (token is truly broken)
        - Uses ReauthCoordinator for interactive OAuth
        - No automatic retry (requires user action)
        - Cleans up unavailable status when done
        """
        # lib_logger.info("Re-auth queue processor started")
        while True:
            path = None
            try:
                # Wait for an item with timeout to allow graceful shutdown
                try:
                    path = await asyncio.wait_for(
                        self._reauth_queue.get(), timeout=60.0
                    )
                except asyncio.TimeoutError:
                    # Queue is empty and idle for 60s - exit
                    self._reauth_processor_task = None
                    # lib_logger.debug("Re-auth queue processor idle, shutting down")
                    return

                try:
                    lib_logger.info(f"Starting re-auth for '{Path(path).name}'...")
                    await self.initialize_token(path, force_interactive=True)
                    lib_logger.info(f"Re-auth SUCCESS for '{Path(path).name}'")

                except Exception as e:
                    lib_logger.error(f"Re-auth FAILED for '{Path(path).name}': {e}")
                    # No automatic retry for re-auth (requires user action)

                finally:
                    # Always clean up
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)
                        self._unavailable_credentials.pop(path, None)
                        # lib_logger.debug(
                        #     f"Re-auth cleanup for '{Path(path).name}'. "
                        #     f"Remaining unavailable: {len(self._unavailable_credentials)}"
                        # )
                    self._reauth_queue.task_done()

            except asyncio.CancelledError:
                # Clean up current credential before breaking
                if path:
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)
                        self._unavailable_credentials.pop(path, None)
                # lib_logger.debug("Re-auth queue processor cancelled")
                break
            except Exception as e:
                lib_logger.error(f"Error in re-auth queue processor: {e}")
                if path:
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)
                        self._unavailable_credentials.pop(path, None)
