"""File-related LangChain tool bundle."""

from __future__ import annotations

from typing import Any

from langchain.tools import tool


def build_file_tools(*, runtime: Any) -> dict[str, Any]:
    """Build uploaded-file and Telegram send tools."""

    @tool
    async def uploaded_file_search(query: str, customer_id: str, limit: int = 5) -> Any:
        """Search uploaded files for this user by natural-language query."""
        safe_limit = max(1, min(int(limit), 20))
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/files/search",
            json_body={
                "query": query,
                "customer_id": customer_id,
                "limit": safe_limit,
            },
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"uploaded_file_search failed: {r.text}"}
        return r.json().get("results", [])

    @tool
    async def uploaded_file_get(
        file_id: str,
        customer_id: str,
        max_excerpt_chars: int = 16000,
    ) -> Any:
        """Get metadata and text excerpt for one uploaded file."""
        safe_chars = max(500, min(int(max_excerpt_chars), 60000))
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/files/get",
            json_body={
                "file_id": file_id,
                "customer_id": customer_id,
                "max_excerpt_chars": safe_chars,
            },
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"uploaded_file_get failed: {r.text}"}
        return r.json().get("file", {})

    @tool
    async def uploaded_file_send(
        file_id: str,
        customer_id: str,
        caption: str | None = None,
    ) -> Any:
        """Send a previously uploaded file back to the user's Telegram chat."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/files/send",
            json_body={
                "file_id": file_id,
                "customer_id": customer_id,
                "caption": caption,
            },
            timeout=25.0,
        )
        if r.status_code != 200:
            return {"error": f"uploaded_file_send failed: {r.text}"}
        return r.json()

    @tool
    async def tulpa_file_send(
        path: str,
        customer_id: str,
        caption: str | None = None,
    ) -> Any:
        """Send a local file from tulpa_stuff/ back to the user's Telegram chat."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/files/send_local",
            json_body={
                "path": path,
                "customer_id": customer_id,
                "caption": caption,
            },
            timeout=25.0,
        )
        if r.status_code != 200:
            return {"error": f"tulpa_file_send failed: {r.text}"}
        return r.json()

    @tool
    async def web_image_send(
        url: str,
        customer_id: str,
        caption: str | None = None,
        max_bytes: int = 10_000_000,
    ) -> Any:
        """
        Download an image from a web URL (validated content-type) and send it to Telegram.
        Use web_search first to find candidate URLs, then call this tool.
        """
        safe_max_bytes = max(250_000, min(int(max_bytes), 25_000_000))
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/files/send_web_image",
            json_body={
                "url": url,
                "customer_id": customer_id,
                "caption": caption,
                "max_bytes": safe_max_bytes,
            },
            timeout=70.0,
            retries=1,
        )
        if r.status_code != 200:
            return {"error": f"web_image_send failed: {r.text}"}
        return r.json()

    @tool
    async def uploaded_file_analyze(
        file_id: str,
        customer_id: str,
        question: str | None = None,
    ) -> Any:
        """Analyze a previously uploaded file again, optionally with a focused question."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/files/analyze",
            json_body={
                "file_id": file_id,
                "customer_id": customer_id,
                "question": question,
            },
            timeout=60.0,
            retries=1,
        )
        if r.status_code != 200:
            return {"error": f"uploaded_file_analyze failed: {r.text}"}
        return r.json()

    return {
        "uploaded_file_search": uploaded_file_search,
        "uploaded_file_get": uploaded_file_get,
        "uploaded_file_send": uploaded_file_send,
        "tulpa_file_send": tulpa_file_send,
        "web_image_send": web_image_send,
        "uploaded_file_analyze": uploaded_file_analyze,
    }
