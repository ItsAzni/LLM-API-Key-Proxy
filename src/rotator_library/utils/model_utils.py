# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/utils/model_utils.py
"""
Shared model string parsing utilities.

Eliminates duplication of provider/model string extraction across
client.py, usage_manager.py, and provider modules.
"""

from __future__ import annotations


def extract_provider_from_model(model: str) -> str:
    """
    Extract provider prefix from ``provider/model`` format.

    Args:
        model: Model string, optionally with ``provider/`` prefix.

    Returns:
        Lowercased provider name, or empty string if no prefix.
    """
    if not isinstance(model, str):
        return ""
    normalized = model.strip()
    if not normalized or "/" not in normalized:
        return ""
    return normalized.split("/", 1)[0].strip().lower()


def normalize_model_string(model: str) -> str:
    """
    Normalize incoming model string for consistent routing.

    Args:
        model: Raw model string from request.

    Returns:
        Stripped model string, or empty string if not a string.
    """
    if not isinstance(model, str):
        return ""
    return model.strip()
