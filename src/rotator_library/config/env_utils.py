# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import os


_TRUTHY_VALUES = frozenset({"true", "1", "yes", "on"})


def env_bool(key: str, default: bool = False) -> bool:
    """Get a boolean from an environment variable."""
    return os.getenv(key, str(default).lower()).lower() in _TRUTHY_VALUES


def env_int(key: str, default: int) -> int:
    """Get an integer from an environment variable, falling back to default."""
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def env_float(key: str, default: float) -> float:
    """Get a float from an environment variable, falling back to default."""
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default
