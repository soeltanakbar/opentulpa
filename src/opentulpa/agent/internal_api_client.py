"""HTTP client wrapper for calling OpenTulpa internal APIs with backoff."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx


class InternalApiClient:
    def __init__(self, *, base_url: str) -> None:
        self.base_url = str(base_url or "").rstrip("/")

    async def request_with_backoff(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout: float = 20.0,
        retries: int = 2,
    ) -> httpx.Response:
        retryable_status = {429, 500, 502, 503, 504}
        retryable_errors = (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
        )
        for attempt in range(retries + 1):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.request(
                        method=method,
                        url=f"{self.base_url}{path}",
                        params=params,
                        json=json_body,
                        timeout=timeout,
                    )
                if response.status_code in retryable_status and attempt < retries:
                    await asyncio.sleep(0.6 * (2**attempt))
                    continue
                return response
            except retryable_errors:
                if attempt < retries:
                    await asyncio.sleep(0.6 * (2**attempt))
                    continue
                raise
        raise RuntimeError("request retry loop exhausted")
