# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/providers/utilities/stealth_client.py
"""
Stealth HTTP client with TLS fingerprinting for Chrome impersonation.

This module provides a drop-in replacement for httpx.AsyncClient that uses
curl_cffi under the hood to produce Chrome-identical TLS ClientHello signatures.

curl_cffi is a Python binding for libcurl-impersonate, which patches libcurl
to produce the exact TLS fingerprint of major browsers (Chrome, Firefox, Safari).

Why this matters:
- Servers can detect bots by analyzing TLS ClientHello signatures (JA3/JA4)
- Python's httpx/requests produce a distinctive Python TLS signature
- Real Chrome browsers have a specific signature that servers expect
- curl_cffi with Chrome impersonation produces byte-identical TLS signatures

Usage:
    # Instead of httpx.AsyncClient
    async with StealthAsyncClient() as client:
        resp = await client.get("https://example.com")

    # With custom impersonate profile
    async with StealthAsyncClient(impersonate="chrome124") as client:
        resp = await client.post("https://api.example.com", json=data)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

lib_logger = logging.getLogger("rotator_library")

CURL_CFFI_AVAILABLE = False
try:
    from curl_cffi.requests import AsyncSession
    from curl_cffi.const import CurlHttpVersion

    CURL_CFFI_AVAILABLE = True
except ImportError:
    lib_logger.warning(
        "curl_cffi not installed. TLS fingerprinting disabled. "
        "Install with: pip install curl_cffi"
    )


CHROME_IMPERSONATE_VERSIONS = [
    "chrome99",
    "chrome100",
    "chrome101",
    "chrome104",
    "chrome107",
    "chrome110",
    "chrome116",
    "chrome119",
    "chrome120",
    "chrome123",
    "chrome124",
    "chrome126",
    "chrome127",
    "chrome131",
    "chrome133",
    "chrome136",
]

DEFAULT_CHROME_VERSION = "chrome136"


class StealthAsyncClient:
    """
    Async HTTP client with Chrome TLS fingerprinting.

    Uses curl_cffi under the hood to produce Chrome-identical TLS signatures.
    Falls back to httpx if curl_cffi is not available.

    Args:
        impersonate: Chrome version to impersonate (e.g., "chrome136")
        timeout: Request timeout in seconds
        headers: Default headers for all requests
        proxies: Proxy configuration (dict with http/https keys)
        verify: Whether to verify SSL certificates
        http2: Enable HTTP/2 support
    """

    def __init__(
        self,
        impersonate: str = DEFAULT_CHROME_VERSION,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
        proxies: Optional[Dict[str, str]] = None,
        verify: bool = True,
        http2: bool = True,
    ):
        self._impersonate = impersonate
        self._timeout = timeout
        self._default_headers = headers or {}
        self._proxies = proxies
        self._verify = verify
        self._http2 = http2
        self._session: Optional[Any] = None
        self._closed = False

    async def __aenter__(self) -> "StealthAsyncClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def _ensure_session(self) -> None:
        """Ensure the underlying session is created."""
        if self._session is None and not self._closed:
            if CURL_CFFI_AVAILABLE:
                self._session = AsyncSession(
                    impersonate=self._impersonate,
                    verify=self._verify,
                    proxies=self._proxies,
                    http_version=CurlHttpVersion.V2_0
                    if self._http2
                    else CurlHttpVersion.V1_1,
                )
            else:
                import httpx

                self._session = httpx.AsyncClient(
                    timeout=self._timeout,
                    verify=self._verify,
                    proxies=self._proxies,
                    http2=self._http2,
                    headers=self._default_headers,
                )

    async def close(self) -> None:
        """Close the underlying session."""
        if self._session is not None and not self._closed:
            try:
                if hasattr(self._session, "close"):
                    if asyncio.iscoroutinefunction(self._session.close):
                        await self._session.close()
                    else:
                        self._session.close()
            except Exception as e:
                lib_logger.debug(f"Error closing stealth client: {e}")
            finally:
                self._session = None
                self._closed = True

    def _merge_headers(
        self, headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """Merge default headers with request-specific headers."""
        merged = dict(self._default_headers)
        if headers:
            merged.update(headers)
        return merged

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Any] = None,
        data: Optional[Union[Dict[str, Any], str, bytes]] = None,
        content: Optional[bytes] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> "StealthResponse":
        """Send an HTTP request."""
        await self._ensure_session()

        merged_headers = self._merge_headers(headers)
        request_timeout = timeout or self._timeout

        if CURL_CFFI_AVAILABLE:
            resp = await self._session.request(
                method=method,
                url=url,
                headers=merged_headers,
                params=params,
                json=json,
                data=data or content,
                timeout=request_timeout,
                **kwargs,
            )
            return StealthResponse.from_curl_cffi(resp)
        else:
            resp = await self._session.request(
                method=method,
                url=url,
                headers=merged_headers,
                params=params,
                json=json,
                data=data or content,
                timeout=request_timeout,
                **kwargs,
            )
            return StealthResponse.from_httpx(resp)

    async def get(self, url: str, **kwargs) -> "StealthResponse":
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> "StealthResponse":
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs) -> "StealthResponse":
        return await self.request("PUT", url, **kwargs)

    async def delete(self, url: str, **kwargs) -> "StealthResponse":
        return await self.request("DELETE", url, **kwargs)

    async def patch(self, url: str, **kwargs) -> "StealthResponse":
        return await self.request("PATCH", url, **kwargs)

    async def stream(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Any] = None,
        data: Optional[Union[Dict[str, Any], str, bytes]] = None,
        content: Optional[bytes] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        """Stream response content."""
        await self._ensure_session()

        merged_headers = self._merge_headers(headers)
        request_timeout = timeout or self._timeout

        if CURL_CFFI_AVAILABLE:
            resp = await self._session.request(
                method=method,
                url=url,
                headers=merged_headers,
                params=params,
                json=json,
                data=data or content,
                timeout=request_timeout,
                stream=True,
                **kwargs,
            )
            return StealthStreamResponse.from_curl_cffi(resp)
        else:
            async with self._session.stream(
                method=method,
                url=url,
                headers=merged_headers,
                params=params,
                json=json,
                data=data or content,
                timeout=request_timeout,
                **kwargs,
            ) as resp:
                return StealthStreamResponse.from_httpx(resp)


class StealthResponse:
    """
    Unified response object wrapping curl_cffi or httpx responses.
    """

    def __init__(
        self,
        status_code: int,
        headers: Dict[str, str],
        content: bytes,
        url: str,
    ):
        self.status_code = status_code
        self.headers = headers
        self._content = content
        self.url = url

    @classmethod
    def from_curl_cffi(cls, resp: Any) -> "StealthResponse":
        """Create from curl_cffi response."""
        return cls(
            status_code=resp.status_code,
            headers=dict(resp.headers),
            content=resp.content,
            url=str(resp.url),
        )

    @classmethod
    def from_httpx(cls, resp: Any) -> "StealthResponse":
        """Create from httpx response."""
        return cls(
            status_code=resp.status_code,
            headers=dict(resp.headers),
            content=resp.content,
            url=str(resp.url),
        )

    @property
    def text(self) -> str:
        """Return response content as text."""
        return self._content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        """Parse response content as JSON."""
        import json

        return json.loads(self._content)

    @property
    def content(self) -> bytes:
        """Return raw response content."""
        return self._content

    def raise_for_status(self) -> None:
        """Raise an exception for HTTP errors."""
        if 400 <= self.status_code < 600:
            raise Exception(f"HTTP {self.status_code} error for {self.url}")


class StealthStreamResponse:
    """
    Unified streaming response object.
    """

    def __init__(
        self,
        status_code: int,
        headers: Dict[str, str],
        url: str,
        iter_bytes: Any,
        close_fn: Optional[callable] = None,
    ):
        self.status_code = status_code
        self.headers = headers
        self.url = url
        self._iter_bytes = iter_bytes
        self._close_fn = close_fn

    @classmethod
    def from_curl_cffi(cls, resp: Any) -> "StealthStreamResponse":
        """Create from curl_cffi streaming response."""
        return cls(
            status_code=resp.status_code,
            headers=dict(resp.headers),
            url=str(resp.url),
            iter_bytes=resp.aiter_content(),
            close_fn=lambda: resp.close() if hasattr(resp, "close") else None,
        )

    @classmethod
    def from_httpx(cls, resp: Any) -> "StealthStreamResponse":
        """Create from httpx streaming response."""
        return cls(
            status_code=resp.status_code,
            headers=dict(resp.headers),
            url=str(resp.url),
            iter_bytes=resp.aiter_bytes(),
            close_fn=None,
        )

    async def aiter_bytes(self) -> bytes:
        """Async iterate over response bytes."""
        async for chunk in self._iter_bytes:
            yield chunk

    async def aread(self) -> bytes:
        """Read the entire response body."""
        chunks = []
        async for chunk in self._iter_bytes:
            chunks.append(chunk)
        return b"".join(chunks)

    async def close(self) -> None:
        """Close the response."""
        if self._close_fn:
            try:
                self._close_fn()
            except Exception:
                pass


def get_available_chrome_versions() -> List[str]:
    """Return list of available Chrome impersonation versions."""
    if CURL_CFFI_AVAILABLE:
        return CHROME_IMPERSONATE_VERSIONS
    return []


def is_tls_fingerprinting_available() -> bool:
    """Check if TLS fingerprinting (curl_cffi) is available."""
    return CURL_CFFI_AVAILABLE
