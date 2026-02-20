"""Tool registration for the OpenTulpa LangGraph runtime."""

from __future__ import annotations

import asyncio
import os
import re
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain.tools import tool

from opentulpa.agent.file_analysis import extract_docx_text, extract_pdf_text
from opentulpa.agent.lc_messages import HumanMessage, SystemMessage
from opentulpa.agent.utils import (
    content_to_text as _content_to_text,
)
from opentulpa.agent.utils import (
    extract_html_title as _extract_html_title,
)
from opentulpa.agent.utils import (
    html_to_text as _html_to_text,
)
from opentulpa.agent.utils import (
    looks_like_shell_command as _looks_like_shell_command,
)


def _browser_use_api_key() -> str:
    return str(os.environ.get("BROWSER_USE_API_KEY", "")).strip()


def _browser_use_base_url() -> str:
    raw = str(os.environ.get("BROWSER_USE_BASE_URL", "")).strip().rstrip("/")
    return raw or "https://api.browser-use.com/api/v2"


def _browser_use_error_detail(resp: httpx.Response) -> str:
    try:
        payload = resp.json()
    except Exception:
        return (resp.text or "").strip()[:500] or f"HTTP {resp.status_code}"
    if isinstance(payload, dict):
        for key in ("detail", "message", "error"):
            value = payload.get(key)
            if value:
                return str(value)
    return str(payload)[:500]


def _normalize_allowed_domains(allowed_domains: list[str] | None) -> list[str]:
    if not isinstance(allowed_domains, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in allowed_domains:
        raw = str(item or "").strip().lower()
        if not raw:
            continue
        host = ""
        if "://" in raw:
            host = str(urlparse(raw).hostname or "").strip().lower()
        else:
            host = raw.split("/", 1)[0].split(":", 1)[0].strip().lower()
        host = host.strip(".")
        if not host or "." not in host:
            continue
        if not re.fullmatch(r"[a-z0-9.-]{1,253}", host):
            continue
        if host in seen:
            continue
        seen.add(host)
        out.append(host)
    return out


def _normalize_cleanup_paths(paths: list[str] | None) -> list[str]:
    if not isinstance(paths, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in paths:
        path = str(item or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _compact_browser_use_task_view(
    payload: dict[str, Any],
    *,
    include_steps: bool = False,
    max_steps_preview: int = 3,
    max_output_chars: int = 12000,
) -> dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    steps = data.get("steps", [])
    steps_list = steps if isinstance(steps, list) else []

    output_text = data.get("output")
    output = str(output_text) if output_text is not None else None
    truncated_output = False
    if output and len(output) > max_output_chars:
        output = output[:max_output_chars] + "..."
        truncated_output = True

    output_files_raw = data.get("outputFiles", [])
    output_files: list[dict[str, Any]] = []
    if isinstance(output_files_raw, list):
        for item in output_files_raw[:20]:
            if isinstance(item, dict):
                output_files.append(
                    {
                        "id": item.get("id"),
                        "fileName": item.get("fileName"),
                    }
                )

    result: dict[str, Any] = {
        "id": data.get("id"),
        "session_id": data.get("sessionId"),
        "status": data.get("status"),
        "is_success": data.get("isSuccess"),
        "started_at": data.get("startedAt"),
        "finished_at": data.get("finishedAt"),
        "task": data.get("task"),
        "llm": data.get("llm"),
        "output": output,
        "output_truncated": truncated_output,
        "output_files": output_files,
        "steps_count": len(steps_list),
    }

    if include_steps:
        safe_preview = max(1, min(int(max_steps_preview), 10))
        preview: list[dict[str, Any]] = []
        for step in steps_list[:safe_preview]:
            if not isinstance(step, dict):
                continue
            actions = step.get("actions", [])
            actions_list = [str(a) for a in actions][:5] if isinstance(actions, list) else []
            preview.append(
                {
                    "number": step.get("number"),
                    "url": step.get("url"),
                    "next_goal": str(step.get("nextGoal") or "")[:240],
                    "actions": actions_list,
                    "screenshot_url": step.get("screenshotUrl"),
                }
            )
        result["steps_preview"] = preview
        result["steps_preview_truncated"] = len(steps_list) > safe_preview
    return result


def register_runtime_tools(runtime: Any) -> dict[str, Any]:
    @tool
    async def memory_search(query: str, customer_id: str) -> Any:
        """Search user memory."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/memory/search",
            json_body={"query": query, "user_id": customer_id, "limit": 5},
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"memory_search failed: {r.text}"}
        return r.json().get("results", [])

    @tool
    async def memory_add(summary: str, customer_id: str) -> Any:
        """Store a user memory summary."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/memory/add",
            json_body={
                "messages": [{"role": "user", "content": summary}],
                "user_id": customer_id,
            },
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"memory_add failed: {r.text}"}
        return {"ok": True}

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

    @tool
    async def skill_list(customer_id: str, include_global: bool = True, limit: int = 50) -> Any:
        """List reusable skills available to this user."""
        safe_limit = max(1, min(int(limit), 200))
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/skills/list",
            json_body={
                "customer_id": customer_id,
                "include_global": bool(include_global),
                "include_disabled": False,
                "limit": safe_limit,
            },
            timeout=8.0,
        )
        if r.status_code != 200:
            return {"error": f"skill_list failed: {r.text}"}
        return r.json().get("skills", [])

    @tool
    async def skill_get(
        name: str,
        customer_id: str,
        include_files: bool = True,
        include_global: bool = True,
    ) -> Any:
        """Get one skill by name, using user-scope first then global fallback."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/skills/get",
            json_body={
                "customer_id": customer_id,
                "name": name,
                "include_files": bool(include_files),
                "include_global": bool(include_global),
            },
            timeout=8.0,
        )
        if r.status_code != 200:
            return {"error": f"skill_get failed: {r.text}"}
        return r.json().get("skill", {})

    @tool
    async def skill_upsert(
        name: str,
        description: str,
        instructions: str,
        customer_id: str,
        scope: str = "user",
        supporting_files: dict[str, str] | None = None,
    ) -> Any:
        """Create or update a reusable skill for this user (or global when explicitly chosen)."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/skills/upsert",
            json_body={
                "customer_id": customer_id,
                "scope": scope,
                "name": name,
                "description": description,
                "instructions": instructions,
                "supporting_files": supporting_files if isinstance(supporting_files, dict) else None,
                "source": "langgraph_tool",
            },
            timeout=20.0,
        )
        if r.status_code != 200:
            return {"error": f"skill_upsert failed: {r.text}"}
        return r.json().get("skill", {})

    @tool
    async def skill_delete(name: str, customer_id: str, scope: str = "user") -> Any:
        """Delete a reusable skill by name."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/skills/delete",
            json_body={
                "customer_id": customer_id,
                "scope": scope,
                "name": name,
            },
            timeout=8.0,
        )
        if r.status_code != 200:
            return {"error": f"skill_delete failed: {r.text}"}
        return r.json()

    @tool
    async def directive_get(customer_id: str) -> Any:
        """Get the active persistent directive profile for this user."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/directive/get",
            json_body={"customer_id": customer_id},
            timeout=5.0,
        )
        if r.status_code != 200:
            return {"error": f"directive_get failed: {r.text}"}
        return r.json()

    @tool
    async def directive_set(directive: str, customer_id: str) -> Any:
        """Set or overwrite the user's persistent directive profile."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/directive/set",
            json_body={
                "customer_id": customer_id,
                "directive": directive,
                "source": "langgraph_tool",
            },
            timeout=5.0,
        )
        if r.status_code != 200:
            return {"error": f"directive_set failed: {r.text}"}
        return r.json()

    @tool
    async def directive_clear(customer_id: str) -> Any:
        """Clear the user's persistent directive profile."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/directive/clear",
            json_body={"customer_id": customer_id},
            timeout=5.0,
        )
        if r.status_code != 200:
            return {"error": f"directive_clear failed: {r.text}"}
        return r.json()

    @tool
    async def time_profile_get(customer_id: str) -> Any:
        """Get stored user UTC offset (if known)."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/time_profile/get",
            json_body={"customer_id": customer_id},
            timeout=5.0,
        )
        if r.status_code != 200:
            return {"error": f"time_profile_get failed: {r.text}"}
        return r.json()

    @tool
    async def time_profile_set(utc_offset: str, customer_id: str) -> Any:
        """Set user UTC offset in +HH:MM or -HH:MM format."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/time_profile/set",
            json_body={
                "customer_id": customer_id,
                "utc_offset": utc_offset,
                "source": "langgraph_tool",
            },
            timeout=5.0,
        )
        if r.status_code != 200:
            return {"error": f"time_profile_set failed: {r.text}"}
        return r.json()

    @tool
    async def web_search(query: str) -> Any:
        """Search the web for current information."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/web_search",
            json_body={"query": query},
            timeout=90.0,
        )
        if r.status_code != 200:
            return {"error": "web_search request failed"}
        return r.json().get("result", "No result.")

    @tool
    async def browser_use_run(
        task: str,
        customer_id: str,
        allowed_domains: list[str] | None = None,
        max_steps: int = 20,
        wait_timeout_seconds: int = 600,
        poll_interval_seconds: int = 4,
        llm: str = "browser-use-llm",
        start_url: str | None = None,
        session_id: str | None = None,
    ) -> Any:
        """
        Run a Browser Use Cloud task and wait for completion.
        Use for dynamic web tasks that need real browser interactions.
        """
        api_key = _browser_use_api_key()
        if not api_key:
            return {"error": "browser_use_run unavailable: BROWSER_USE_API_KEY missing"}

        task_text = str(task or "").strip()
        if not task_text:
            return {"error": "browser_use_run requires a non-empty task"}

        safe_max_steps = max(1, min(int(max_steps), 80))
        safe_wait_timeout = max(30, min(int(wait_timeout_seconds), 1800))
        safe_poll_interval = max(2, min(int(poll_interval_seconds), 30))
        safe_domains = _normalize_allowed_domains(allowed_domains)
        safe_llm = str(llm or "").strip() or "browser-use-llm"
        safe_start_url = str(start_url or "").strip()
        safe_session_id = str(session_id or "").strip()

        payload: dict[str, Any] = {
            "task": task_text,
            "maxSteps": safe_max_steps,
            "llm": safe_llm,
            "metadata": {
                "source": "opentulpa",
                "customer_id": str(customer_id or "").strip()[:120],
            },
        }
        if safe_domains:
            payload["allowedDomains"] = safe_domains
        if safe_start_url:
            payload["startUrl"] = safe_start_url
        if safe_session_id:
            payload["sessionId"] = safe_session_id

        headers = {"X-Browser-Use-API-Key": api_key, "Content-Type": "application/json"}
        timeout = httpx.Timeout(60.0, connect=10.0, read=60.0)
        base_url = _browser_use_base_url()
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers=headers,
        ) as client:
            try:
                create_resp = await client.post(f"{base_url}/tasks", json=payload)
            except Exception as exc:
                return {"error": f"browser_use_run create request failed: {exc}"}
            if create_resp.status_code not in {200, 201, 202}:
                return {
                    "error": (
                        f"browser_use_run create failed: HTTP {create_resp.status_code}: "
                        f"{_browser_use_error_detail(create_resp)}"
                    )
                }
            try:
                created = create_resp.json()
            except Exception:
                return {"error": "browser_use_run create failed: invalid JSON response"}

            task_id = str(created.get("id") or "").strip()
            result_session_id = str(created.get("sessionId") or safe_session_id).strip()
            if not task_id:
                return {"error": "browser_use_run create failed: missing task id in response"}

            deadline = datetime.now(timezone.utc).timestamp() + safe_wait_timeout
            while True:
                try:
                    task_resp = await client.get(f"{base_url}/tasks/{task_id}")
                except Exception as exc:
                    return {"error": f"browser_use_run poll failed: {exc}", "task_id": task_id}
                if task_resp.status_code != 200:
                    return {
                        "error": (
                            f"browser_use_run poll failed: HTTP {task_resp.status_code}: "
                            f"{_browser_use_error_detail(task_resp)}"
                        ),
                        "task_id": task_id,
                        "session_id": result_session_id or None,
                    }
                try:
                    task_data = task_resp.json()
                except Exception:
                    return {"error": "browser_use_run poll failed: invalid JSON", "task_id": task_id}

                status = str(task_data.get("status") or "").strip().lower()
                if status in {"finished", "stopped"}:
                    live_url = None
                    if result_session_id:
                        with_context = await client.get(f"{base_url}/sessions/{result_session_id}")
                        if with_context.status_code == 200:
                            with suppress(Exception):
                                live_url = with_context.json().get("liveUrl")
                    compact = _compact_browser_use_task_view(task_data)
                    compact["task_id"] = task_id
                    compact["session_id"] = result_session_id or compact.get("session_id")
                    compact["status"] = status or str(compact.get("status") or "unknown")
                    compact["live_url"] = live_url
                    return compact

                if datetime.now(timezone.utc).timestamp() >= deadline:
                    return {
                        "task_id": task_id,
                        "session_id": result_session_id or None,
                        "status": status or "started",
                        "timed_out": True,
                        "message": (
                            "Task is still running. Use browser_use_task_get(task_id) "
                            "to check progress or browser_use_task_control to stop."
                        ),
                    }

                await asyncio.sleep(safe_poll_interval)

    @tool
    async def browser_use_task_get(
        task_id: str,
        include_steps: bool = False,
        max_steps_preview: int = 3,
    ) -> Any:
        """Get Browser Use task status/details by task_id (compact by default)."""
        api_key = _browser_use_api_key()
        if not api_key:
            return {"error": "browser_use_task_get unavailable: BROWSER_USE_API_KEY missing"}

        safe_task_id = str(task_id or "").strip()
        if not safe_task_id:
            return {"error": "browser_use_task_get requires task_id"}

        headers = {"X-Browser-Use-API-Key": api_key}
        timeout = httpx.Timeout(45.0, connect=10.0, read=45.0)
        base_url = _browser_use_base_url()
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers=headers,
        ) as client:
            try:
                resp = await client.get(f"{base_url}/tasks/{safe_task_id}")
            except Exception as exc:
                return {"error": f"browser_use_task_get request failed: {exc}"}
            if resp.status_code != 200:
                return {
                    "error": (
                        f"browser_use_task_get failed: HTTP {resp.status_code}: "
                        f"{_browser_use_error_detail(resp)}"
                    )
                }
            try:
                payload = resp.json()
            except Exception:
                return {"error": "browser_use_task_get failed: invalid JSON response"}
            return _compact_browser_use_task_view(
                payload if isinstance(payload, dict) else {},
                include_steps=bool(include_steps),
                max_steps_preview=max_steps_preview,
            )

    @tool
    async def browser_use_task_control(task_id: str, action: str = "stop_task_and_session") -> Any:
        """Control Browser Use task execution (stop, pause, resume, or stop_task_and_session)."""
        api_key = _browser_use_api_key()
        if not api_key:
            return {"error": "browser_use_task_control unavailable: BROWSER_USE_API_KEY missing"}

        safe_task_id = str(task_id or "").strip()
        if not safe_task_id:
            return {"error": "browser_use_task_control requires task_id"}
        safe_action = str(action or "").strip().lower()
        allowed_actions = {"stop", "pause", "resume", "stop_task_and_session"}
        if safe_action not in allowed_actions:
            return {
                "error": (
                    "browser_use_task_control invalid action. "
                    "Use one of: stop, pause, resume, stop_task_and_session"
                )
            }

        headers = {"X-Browser-Use-API-Key": api_key, "Content-Type": "application/json"}
        timeout = httpx.Timeout(45.0, connect=10.0, read=45.0)
        base_url = _browser_use_base_url()
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers=headers,
        ) as client:
            try:
                resp = await client.patch(
                    f"{base_url}/tasks/{safe_task_id}",
                    json={"action": safe_action},
                )
            except Exception as exc:
                return {"error": f"browser_use_task_control request failed: {exc}"}
            if resp.status_code != 200:
                return {
                    "error": (
                        f"browser_use_task_control failed: HTTP {resp.status_code}: "
                        f"{_browser_use_error_detail(resp)}"
                    )
                }
            try:
                payload = resp.json()
            except Exception:
                return {"error": "browser_use_task_control failed: invalid JSON response"}
            return _compact_browser_use_task_view(payload if isinstance(payload, dict) else {})

    @tool
    async def fetch_link_content(
        url: str,
        max_chars: int = 40000,
        use_vision_for_images: bool = True,
    ) -> Any:
        """
        Fetch and extract content from a URL.
        Supports HTML/text/JSON, PDF, DOCX, and images (via model vision).
        """
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

        try:
            if ctype.startswith("image/"):
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
            elif ctype == "application/pdf" or final_url.lower().endswith(".pdf"):
                mode = "pdf"
                text_content = extract_pdf_text(resp.content)
            elif (
                ctype == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                or final_url.lower().endswith(".docx")
            ):
                mode = "docx"
                text_content = extract_docx_text(resp.content)
            else:
                mode = "web_text"
                raw_text = resp.text
                if "html" in ctype or "<html" in raw_text.lower():
                    title = _extract_html_title(raw_text)
                    text_content = _html_to_text(raw_text)
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
    async def tulpa_write_file(path: str, content: str) -> Any:
        """Write file in approved paths."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/tulpa/write_file",
            json_body={"path": path, "content": content},
            timeout=20.0,
        )
        if r.status_code != 200:
            return {"error": f"write failed: {r.text}"}
        return r.json()

    @tool
    async def tulpa_validate_file(path: str) -> Any:
        """Validate generated file syntax/contracts in approved paths."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/tulpa/validate_file",
            json_body={"path": path},
            timeout=20.0,
        )
        if r.status_code != 200:
            return {"error": f"validation failed: {r.text}"}
        return r.json()

    @tool
    async def tulpa_run_terminal(
        command: str,
        working_dir: str = "tulpa_stuff",
        timeout_seconds: int = 90,
    ) -> Any:
        """Run approved terminal command."""
        if not _looks_like_shell_command(command):
            return {
                "error": (
                    "Command rejected: provide a concrete shell command (executable + args), "
                    "not natural language."
                )
            }
        safe_timeout = max(5, min(int(timeout_seconds), 600))
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/tulpa/run_terminal",
            json_body={
                "command": command,
                "working_dir": working_dir,
                "timeout_seconds": safe_timeout,
            },
            timeout=max(10.0, float(safe_timeout) + 10.0),
            retries=1,
        )
        if r.status_code != 200:
            return {"error": f"terminal failed: {r.text}"}
        return r.json()

    @tool
    async def tulpa_read_file(path: str, max_chars: int = 12000) -> Any:
        """Read file in approved paths."""
        safe_max_chars = max(500, min(int(max_chars), 20000))
        r = await runtime._request_with_backoff(
            "GET",
            "/internal/tulpa/read_file",
            params={"path": path, "max_chars": safe_max_chars},
            timeout=15.0,
        )
        if r.status_code != 200:
            return {"error": f"read failed: {r.text}"}
        return r.json()

    @tool
    async def tulpa_catalog() -> Any:
        """Get catalog of tracked files and artifacts."""
        r = await runtime._request_with_backoff("GET", "/internal/tulpa/catalog", timeout=10.0)
        if r.status_code != 200:
            return {"error": f"catalog failed: {r.text}"}
        return r.json().get("catalog", {})

    @tool
    async def task_status(task_id: str) -> Any:
        """Get task status."""
        r = await runtime._request_with_backoff("GET", f"/internal/tasks/{task_id}", timeout=10.0)
        if r.status_code != 200:
            return {"error": f"task_status failed: {r.text}"}
        return r.json().get("task", {})

    @tool
    async def task_events(task_id: str, limit: int = 30, offset: int = 0) -> Any:
        """Get task events."""
        r = await runtime._request_with_backoff(
            "GET",
            f"/internal/tasks/{task_id}/events",
            params={"limit": max(1, min(int(limit), 200)), "offset": max(0, int(offset))},
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"task_events failed: {r.text}"}
        return r.json().get("events", [])

    @tool
    async def task_artifacts(task_id: str) -> Any:
        """Get task artifacts."""
        r = await runtime._request_with_backoff(
            "GET", f"/internal/tasks/{task_id}/artifacts", timeout=10.0
        )
        if r.status_code != 200:
            return {"error": f"task_artifacts failed: {r.text}"}
        return r.json().get("artifacts", [])

    @tool
    async def task_relaunch(
        task_id: str, clarification: str | None = None, trigger_reason: str = "user_requested"
    ) -> Any:
        """Relaunch a task."""
        r = await runtime._request_with_backoff(
            "POST",
            f"/internal/tasks/{task_id}/relaunch",
            json_body={"clarification": clarification, "trigger_reason": trigger_reason},
            timeout=20.0,
        )
        if r.status_code != 200:
            return {"error": f"task_relaunch failed: {r.text}"}
        return r.json().get("task", {})

    @tool
    async def task_cancel(task_id: str) -> Any:
        """Cancel a task."""
        r = await runtime._request_with_backoff(
            "POST", f"/internal/tasks/{task_id}/cancel", timeout=10.0
        )
        if r.status_code != 200:
            return {"error": f"task_cancel failed: {r.text}"}
        return r.json().get("task", {})

    @tool
    async def routine_create(
        name: str,
        schedule: str,
        message: str,
        customer_id: str,
        notify_user: bool = True,
        cleanup_paths: list[str] | None = None,
    ) -> Any:
        """
        Create a scheduled routine.
        - Recurring: cron (e.g. "0 9 * * *")
        - One-time: local ISO datetime (e.g. "2026-02-18T23:45:00+08:00")
        - cleanup_paths: optional repo-relative file paths to remove when deleting this automation.
        """
        auto_notify = bool(notify_user)
        safe_cleanup_paths = _normalize_cleanup_paths(cleanup_paths)

        r = await runtime._request_with_backoff(
            "POST",
            "/internal/scheduler/routine",
            json_body={
                "name": name,
                "schedule": schedule,
                "payload": {
                    "message": message,
                    "customer_id": customer_id,
                    "notify_user": auto_notify,
                    "notification_opt_out": not auto_notify,
                    "cleanup_paths": safe_cleanup_paths,
                },
                "is_cron": " " in schedule and len(schedule.split()) >= 5,
            },
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"routine_create failed: {r.text}"}
        return r.json()

    @tool
    async def routine_list(customer_id: str) -> Any:
        """List routines for the current user."""
        r = await runtime._request_with_backoff(
            "GET",
            "/internal/scheduler/routines",
            params={"customer_id": customer_id},
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"routine_list failed: {r.text}"}
        return r.json().get("routines", [])

    @tool
    async def routine_delete(routine_id: str, customer_id: str) -> Any:
        """Delete/stop one routine by id for the current user."""
        rid = str(routine_id or "").strip()
        if not rid:
            return {"error": "routine_delete failed: routine_id is required"}

        r = await runtime._request_with_backoff(
            "DELETE",
            f"/internal/scheduler/routine/{rid}",
            params={"customer_id": customer_id},
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"routine_delete failed: {r.text}"}
        payload = r.json() if r.content else {}
        if not bool(payload.get("ok")):
            return {
                "error": "routine_delete failed: routine not found or not accessible",
                "routine_id": rid,
            }

        verify = await runtime._request_with_backoff(
            "GET",
            "/internal/scheduler/routines",
            params={"customer_id": customer_id},
            timeout=10.0,
        )
        if verify.status_code != 200:
            return {
                "ok": True,
                "routine_id": rid,
                "verified_removed": False,
                "warning": "delete succeeded but verification list failed",
            }
        routines = verify.json().get("routines", [])
        still_present = any(str(item.get("id", "")) == rid for item in routines if isinstance(item, dict))
        return {
            "ok": not still_present,
            "routine_id": rid,
            "verified_removed": not still_present,
            "remaining_routines": routines,
        }

    @tool
    async def automation_delete(
        routine_id: str,
        customer_id: str,
        delete_files: bool = True,
        cleanup_paths: list[str] | None = None,
    ) -> Any:
        """Delete an automation by id, including optional script/file cleanup."""
        rid = str(routine_id or "").strip()
        if not rid:
            return {"error": "automation_delete failed: routine_id is required"}
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/scheduler/routine/delete_with_assets",
            json_body={
                "customer_id": customer_id,
                "routine_id": rid,
                "delete_files": bool(delete_files),
                "cleanup_paths": _normalize_cleanup_paths(cleanup_paths),
            },
            timeout=20.0,
        )
        if r.status_code != 200:
            return {"error": f"automation_delete failed: {r.text}"}
        return r.json()

    @tool
    async def guardrail_execute_approved_action(approval_id: str, customer_id: str) -> Any:
        """Execute a previously approved external-impact action exactly once."""
        aid = str(approval_id or "").strip()
        if not aid:
            return {"error": "guardrail_execute_approved_action requires approval_id"}
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/approvals/execute",
            json_body={"approval_id": aid, "customer_id": customer_id},
            timeout=600.0,
            retries=0,
        )
        if r.status_code != 200:
            return {"error": f"guardrail_execute_approved_action failed: {r.text}"}
        return r.json()

    @tool
    async def server_time() -> Any:
        """Get server time."""
        now_local = datetime.now().astimezone()
        now_utc = datetime.now(timezone.utc)
        return {
            "server_time_local_iso": now_local.isoformat(),
            "server_timezone": str(now_local.tzinfo),
            "server_time_utc_iso": now_utc.isoformat(),
            "unix_timestamp": int(now_utc.timestamp()),
        }

    return {
        "memory_search": memory_search,
        "memory_add": memory_add,
        "uploaded_file_search": uploaded_file_search,
        "uploaded_file_get": uploaded_file_get,
        "uploaded_file_send": uploaded_file_send,
        "web_image_send": web_image_send,
        "uploaded_file_analyze": uploaded_file_analyze,
        "skill_list": skill_list,
        "skill_get": skill_get,
        "skill_upsert": skill_upsert,
        "skill_delete": skill_delete,
        "directive_get": directive_get,
        "directive_set": directive_set,
        "directive_clear": directive_clear,
        "time_profile_get": time_profile_get,
        "time_profile_set": time_profile_set,
        "web_search": web_search,
        "browser_use_run": browser_use_run,
        "browser_use_task_get": browser_use_task_get,
        "browser_use_task_control": browser_use_task_control,
        "fetch_link_content": fetch_link_content,
        "tulpa_write_file": tulpa_write_file,
        "tulpa_validate_file": tulpa_validate_file,
        "tulpa_run_terminal": tulpa_run_terminal,
        "tulpa_read_file": tulpa_read_file,
        "tulpa_catalog": tulpa_catalog,
        "task_status": task_status,
        "task_events": task_events,
        "task_artifacts": task_artifacts,
        "task_relaunch": task_relaunch,
        "task_cancel": task_cancel,
        "routine_create": routine_create,
        "routine_list": routine_list,
        "routine_delete": routine_delete,
        "automation_delete": automation_delete,
        "guardrail_execute_approved_action": guardrail_execute_approved_action,
        "server_time": server_time,
    }
