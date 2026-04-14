# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
Quota cache serialization helpers for logging.

Shared by BaseQuotaTracker and lightweight trackers to format
quota cache entries for structured logging without duplicating logic.
"""

from typing import Any, Dict


def serialize_quota_cache(cache_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Return a log-safe copy of the quota cache, truncating credential keys."""
    return {
        f"...{key[-4:]}": {
            k: v for k, v in entry.items() if k != "error"
        }
        for key, entry in cache_dict.items()
    }


def format_quota_log_entry(provider_name: str, api_key: str, cache_entry: Dict[str, Any]) -> str:
    """Format a single quota cache entry as a one-line log string."""
    remaining = cache_entry.get("remaining", "?")
    quota = cache_entry.get("quota", "?")
    fraction = cache_entry.get("remaining_fraction", 0)
    pct = f"{fraction * 100:.0f}%" if isinstance(fraction, (int, float)) else "?%"
    return f"{provider_name} ...{api_key[-4:]}: {remaining}/{quota} remaining ({pct})"
