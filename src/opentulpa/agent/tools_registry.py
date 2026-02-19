"""Tool registration for the OpenTulpa LangGraph runtime."""

from __future__ import annotations

import re
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
    ) -> Any:
        """
        Create a scheduled routine.
        - Recurring: cron (e.g. "0 9 * * *")
        - One-time: local ISO datetime (e.g. "2026-02-18T23:45:00+08:00")
        """
        auto_notify = bool(notify_user)

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
                },
                "is_cron": " " in schedule and len(schedule.split()) >= 5,
            },
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"routine_create failed: {r.text}"}
        return r.json()

    @tool
    async def routine_list() -> Any:
        """List routines."""
        r = await runtime._request_with_backoff(
            "GET", "/internal/scheduler/routines", timeout=10.0
        )
        if r.status_code != 200:
            return {"error": f"routine_list failed: {r.text}"}
        return r.json().get("routines", [])

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
        "server_time": server_time,
    }
