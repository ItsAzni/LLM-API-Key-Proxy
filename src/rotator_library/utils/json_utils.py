# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""High-performance JSON serialization utilities using orjson.

orjson is significantly faster than the standard json module for
serialization and deserialization of large payloads in hot paths
(API request/response transformation, streaming, message cloning).
"""

import orjson
from typing import Any


def json_dumps(obj: Any) -> bytes:
    """Fast JSON serialization using orjson. Returns UTF-8 encoded bytes."""
    return orjson.dumps(obj)


def json_dumps_str(obj: Any) -> str:
    """Fast JSON serialization returning a UTF-8 string."""
    return orjson.dumps(obj).decode("utf-8")


def json_loads(s: str) -> Any:
    """Fast JSON deserialization using orjson."""
    return orjson.loads(s)
