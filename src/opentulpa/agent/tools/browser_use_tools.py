"""Browser Use LangChain tool bundle."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any

import httpx
from langchain.tools import tool


def build_browser_use_tools(
    *,
    runtime: Any,
    browser_use_api_key: Callable[[], str],
    browser_use_base_url: Callable[[], str],
    browser_use_error_detail: Callable[[httpx.Response], str],
    normalize_allowed_domains: Callable[[list[str] | None], list[str]],
    compact_browser_use_task_view: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Build Browser Use Cloud task tools."""

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
        api_key = browser_use_api_key()
        if not api_key:
            return {"error": "browser_use_run unavailable: BROWSER_USE_API_KEY missing"}

        task_text = str(task or "").strip()
        if not task_text:
            return {"error": "browser_use_run requires a non-empty task"}

        safe_max_steps = max(1, min(int(max_steps), 80))
        safe_wait_timeout = max(30, min(int(wait_timeout_seconds), 1800))
        safe_poll_interval = max(2, min(int(poll_interval_seconds), 30))
        safe_domains = normalize_allowed_domains(allowed_domains)
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
        base_url = browser_use_base_url()
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
                        f"{browser_use_error_detail(create_resp)}"
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
                            f"{browser_use_error_detail(task_resp)}"
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
                    compact = compact_browser_use_task_view(task_data)
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
        api_key = browser_use_api_key()
        if not api_key:
            return {"error": "browser_use_task_get unavailable: BROWSER_USE_API_KEY missing"}

        safe_task_id = str(task_id or "").strip()
        if not safe_task_id:
            return {"error": "browser_use_task_get requires task_id"}

        headers = {"X-Browser-Use-API-Key": api_key}
        timeout = httpx.Timeout(45.0, connect=10.0, read=45.0)
        base_url = browser_use_base_url()
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
                        f"{browser_use_error_detail(resp)}"
                    )
                }
            try:
                payload = resp.json()
            except Exception:
                return {"error": "browser_use_task_get failed: invalid JSON response"}
            return compact_browser_use_task_view(
                payload if isinstance(payload, dict) else {},
                include_steps=bool(include_steps),
                max_steps_preview=max_steps_preview,
            )

    @tool
    async def browser_use_task_control(task_id: str, action: str = "stop_task_and_session") -> Any:
        """Control Browser Use task execution (stop, pause, resume, or stop_task_and_session)."""
        api_key = browser_use_api_key()
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
        base_url = browser_use_base_url()
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
                        f"{browser_use_error_detail(resp)}"
                    )
                }
            try:
                payload = resp.json()
            except Exception:
                return {"error": "browser_use_task_control failed: invalid JSON response"}
            return compact_browser_use_task_view(payload if isinstance(payload, dict) else {})

    return {
        "browser_use_run": browser_use_run,
        "browser_use_task_get": browser_use_task_get,
        "browser_use_task_control": browser_use_task_control,
    }
