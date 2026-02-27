"""Remote URL/file content extraction LangChain tool bundle."""

from __future__ import annotations

import json
import re
from contextlib import suppress
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain.tools import tool

from opentulpa.agent.file_analysis import summarize_uploaded_blob
from opentulpa.agent.lc_messages import HumanMessage, SystemMessage
from opentulpa.agent.utils import content_to_text as _content_to_text
from opentulpa.agent.utils import extract_html_title as _extract_html_title
from opentulpa.agent.utils import html_to_text as _html_to_text


def _best_crawl4ai_text(result: Any) -> tuple[str, str | None]:
    title: str | None = None
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        raw_title = metadata.get("title")
        if isinstance(raw_title, str) and raw_title.strip():
            title = raw_title.strip()

    candidates = [
        getattr(result, "fit_markdown", None),
        getattr(result, "markdown", None),
        getattr(result, "extracted_content", None),
        getattr(result, "cleaned_html", None),
        getattr(result, "html", None),
        getattr(result, "text", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        text = ""
        if isinstance(candidate, str):
            text = candidate
        elif isinstance(candidate, (dict, list)):
            with suppress(Exception):
                text = json.dumps(candidate, ensure_ascii=False)
        else:
            text = str(candidate)
        text = str(text).strip()
        if not text:
            continue
        if "<html" in text.lower() or "</" in text:
            text = _html_to_text(text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            return text, title
    return "", title


async def _crawl4ai_extract(url: str) -> tuple[str, str | None, str | None]:
    """
    Return (text, title, error). If crawl4ai is unavailable/failed, text is empty and error set.
    """
    try:
        from crawl4ai import AsyncWebCrawler
    except Exception as exc:
        return "", None, f"crawl4ai unavailable: {exc}"

    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
    except Exception as exc:
        return "", None, f"crawl4ai crawl failed: {exc}"

    if bool(getattr(result, "success", True)) is False:
        error_message = str(getattr(result, "error_message", "")).strip()
        return "", None, f"crawl4ai crawl failed: {error_message or 'unknown_error'}"

    text, title = _best_crawl4ai_text(result)
    if not text:
        return "", title, "crawl4ai returned no extractable content"
    return text, title, None


def build_content_fetch_tools(*, runtime: Any) -> dict[str, Any]:
    """Build web/file fetch-and-extract tools."""

    async def _fetch_remote_content(
        url: str,
        max_chars: int = 40000,
        use_vision_for_images: bool = True,
        target: str = "url",
    ) -> Any:
        """Fetch and extract content from URL based on target type (url|file)."""
        raw_url = str(url or "").strip()
        parsed = urlparse(raw_url)
        if parsed.scheme not in {"http", "https"}:
            return {"error": "url must start with http:// or https://"}

        safe_max_chars = max(2000, min(int(max_chars), 120000))
        try:
            async with httpx.AsyncClient(
                timeout=45.0,
                follow_redirects=True,
                headers={"User-Agent": "OpenTulpa/0.1 (+content-fetch)"},
            ) as client:
                resp = await client.get(raw_url)
        except Exception as exc:
            return {"error": f"link fetch failed: {exc}"}

        if resp.status_code >= 400:
            return {"error": f"link fetch failed: HTTP {resp.status_code}"}

        ctype = str(resp.headers.get("content-type", "")).split(";")[0].strip().lower()
        final_url = str(resp.url)
        text_content = ""
        title: str | None = None
        mode = "text"
        is_image = ctype.startswith("image/")
        is_pdf = ctype == "application/pdf" or final_url.lower().endswith(".pdf")
        is_docx = (
            ctype == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            or final_url.lower().endswith(".docx")
        )
        file_like = is_image or is_pdf or is_docx

        safe_target = str(target or "url").strip().lower()
        if safe_target == "url" and file_like:
            return {
                "error": (
                    "URL points to a file-like resource (image/pdf/docx). "
                    "Use fetch_file_content instead."
                ),
                "url": final_url,
                "content_type": ctype or "unknown",
            }
        if safe_target == "file" and not file_like:
            return {
                "error": (
                    "URL does not point to supported file-like content (image/pdf/docx). "
                    "Use fetch_url_content instead."
                ),
                "url": final_url,
                "content_type": ctype or "unknown",
            }

        try:
            if is_image:
                mode = "image_vision"
                if use_vision_for_images:
                    vision = await runtime._model.ainvoke(
                        [
                            SystemMessage(
                                content=(
                                    "Describe the image and extract all readable text. "
                                    "If it is a screenshot/document, summarize key points."
                                )
                            ),
                            HumanMessage(
                                content=[
                                    {"type": "text", "text": "Analyze this image URL."},
                                    {"type": "image_url", "image_url": {"url": final_url}},
                                ]
                            ),
                        ]
                    )
                    text_content = _content_to_text(getattr(vision, "content", "")).strip()
                else:
                    text_content = ""
            elif is_pdf:
                mode = "pdf_llm"
                text_content = await summarize_uploaded_blob(
                    runtime,
                    filename=final_url.rsplit("/", 1)[-1] or "document.pdf",
                    mime_type=ctype or "application/pdf",
                    kind="document",
                    raw_bytes=resp.content,
                    question=(
                        "Extract key information from this PDF and provide a concise but complete "
                        "summary with important facts, entities, dates, and actions."
                    ),
                )
            elif is_docx:
                mode = "docx_llm"
                text_content = await summarize_uploaded_blob(
                    runtime,
                    filename=final_url.rsplit("/", 1)[-1] or "document.docx",
                    mime_type=ctype
                    or "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    kind="document",
                    raw_bytes=resp.content,
                    question=(
                        "Extract key information from this DOCX and provide a concise but complete "
                        "summary with important facts, entities, dates, and actions."
                    ),
                )
            else:
                mode = "web_text"
                raw_text = resp.text
                if "html" in ctype or "<html" in raw_text.lower():
                    crawled_text, crawled_title, crawl_error = await _crawl4ai_extract(final_url)
                    if crawled_text:
                        mode = "web_text_crawl4ai"
                        title = crawled_title or _extract_html_title(raw_text)
                        text_content = crawled_text
                    else:
                        mode = "web_text_fallback"
                        title = _extract_html_title(raw_text)
                        text_content = _html_to_text(raw_text)
                        if crawl_error:
                            text_content = (
                                f"[crawl4ai_fallback_reason] {crawl_error}\n\n{text_content}"
                            )
                else:
                    text_content = raw_text
        except Exception as exc:
            return {
                "error": f"content extraction failed: {exc}",
                "url": final_url,
                "content_type": ctype or "unknown",
            }

        normalized = re.sub(r"\n{3,}", "\n\n", str(text_content or "").strip())
        truncated = len(normalized) > safe_max_chars
        return {
            "url": final_url,
            "content_type": ctype or "unknown",
            "mode": mode,
            "title": title,
            "char_count": len(normalized),
            "truncated": truncated,
            "text": normalized[:safe_max_chars],
        }

    @tool
    async def fetch_url_content(url: str, max_chars: int = 40000) -> Any:
        """Fetch and extract web page/text/JSON content from a URL."""
        return await _fetch_remote_content(
            url=url,
            max_chars=max_chars,
            use_vision_for_images=False,
            target="url",
        )

    @tool
    async def fetch_file_content(
        url: str,
        max_chars: int = 40000,
        use_vision_for_images: bool = True,
    ) -> Any:
        """Fetch and analyze file-like URL content (image/pdf/docx)."""
        return await _fetch_remote_content(
            url=url,
            max_chars=max_chars,
            use_vision_for_images=use_vision_for_images,
            target="file",
        )

    return {
        "fetch_url_content": fetch_url_content,
        "fetch_file_content": fetch_file_content,
    }
