# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""HTTP retry utilities with exponential backoff."""

import random
from typing import Optional


def compute_backoff_with_jitter(
    attempt: int,
    base: float = 2.0,
    max_wait: float = 60.0,
    jitter: float = 0.3,
    retry_after: Optional[float] = None,
    min_wait: float = 0.5,
) -> float:
    """
    Compute exponential backoff wait time with jitter. Returns wait time in seconds.

    Args:
        attempt: Current attempt number (0-indexed)
        base: Base for exponential calculation
        max_wait: Maximum wait time in seconds
        jitter: Jitter factor (0.0-1.0)
        retry_after: Optional override wait time from server
        min_wait: Minimum wait time in seconds (default: 0.5)
    """
    wait_time = min(base**attempt, max_wait)

    if jitter > 0:
        jitter_amount = random.uniform(-jitter, jitter) * wait_time
        wait_time = max(min_wait, wait_time + jitter_amount)

    if retry_after is not None and retry_after > 0:
        wait_time = min(max(wait_time, retry_after, min_wait), max_wait)

    return wait_time


