"""Helper utilities used by tool registry composition."""

from __future__ import annotations

import os
import re
from contextlib import suppress
from typing import Any
from urllib.parse import urlparse

import httpx

from opentulpa.agent.result_models import ToolGuardrailDecision
from opentulpa.policy.execution_boundary import ExecutionBoundaryGuard


def browser_use_api_key() -> str:
    return str(os.environ.get("BROWSER_USE_API_KEY", "")).strip()


def browser_use_base_url() -> str:
    raw = str(os.environ.get("BROWSER_USE_BASE_URL", "")).strip().rstrip("/")
    return raw or "https://api.browser-use.com/api/v2"


def browser_use_error_detail(resp: httpx.Response) -> str:
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


def normalize_allowed_domains(allowed_domains: list[str] | None) -> list[str]:
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


def normalize_cleanup_paths(paths: list[str] | None) -> list[str]:
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


def normalize_execution_origin(
    *,
    thread_id: str | None,
    execution_origin: str | None,
) -> str:
    return ExecutionBoundaryGuard.normalize_execution_origin(
        thread_id=str(thread_id or "").strip(),
        execution_origin=str(execution_origin or "").strip(),
    )


def approval_pending_payload(
    *,
    action_name: str,
    command_preview: str,
    decision: ToolGuardrailDecision,
) -> dict[str, Any]:
    approval_id = str(decision.approval_id or "").strip()
    summary = str(decision.summary or "").strip() or f"execute {action_name}"
    reason = str(decision.reason or "approval_required").strip()
    message = (
        "APPROVAL_PENDING: This executable action is waiting for user approval "
        f"(approval_id={approval_id}; summary={summary}; reason={reason})."
    )
    return {
        "ok": False,
        "status": "approval_pending",
        "action_name": action_name,
        "command_preview": command_preview[:300],
        "approval_id": approval_id or None,
        "delivery_mode": str(decision.delivery_mode or "").strip() or None,
        "summary": summary,
        "reason": reason,
        "message": message,
        "gate": "require_approval",
    }


def compact_browser_use_task_view(
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


def _sanitize_routine_customer_segment(customer_id: str) -> str:
    raw = str(customer_id or "").strip().lower()
    safe = re.sub(r"[^a-z0-9_-]+", "-", raw).strip("-")
    return (safe or "customer")[:48]


def _proactive_heartbeat_routine_id(customer_id: str) -> str:
    return f"rtn_proactive_{_sanitize_routine_customer_segment(customer_id)}"


def _directive_disables_proactive_mode(directive: str) -> bool:
    text = str(directive or "").strip().lower()
    if not text:
        return False
    patterns = [
        r"\b(?:disable|turn off|stop|pause|remove)\s+(?:my\s+)?proactive\b",
        r"\bnot\s+proactive\b",
        r"\bmode\s*[:=]?\s*non[- ]?proactive\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _directive_enables_proactive_mode(directive: str) -> bool:
    text = str(directive or "").strip().lower()
    if not text or _directive_disables_proactive_mode(text):
        return False
    patterns = [
        r"\bmode\s*[:=]?\s*proactive\b",
        r"\bproactive\s+mode\b",
        r"\bproactive\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _extract_heartbeat_interval_hours(directive: str, *, default_hours: int) -> int:
    text = str(directive or "").strip().lower()
    interval = max(1, min(int(default_hours), 24))
    if not text:
        return interval
    match = re.search(r"\bevery\s+(\d{1,2})\s*(?:hours?|hrs?|h)\b", text)
    if match:
        with suppress(Exception):
            return max(1, min(int(match.group(1)), 24))
    if re.search(r"\bevery\s+(?:few)\s+hours?\b", text):
        return 3
    if re.search(r"\bevery\s+(?:couple)\s+hours?\b", text):
        return 2
    return interval


def _build_proactive_heartbeat_prompt(interval_hours: int) -> str:
    return (
        "Proactive heartbeat wake. Decide naturally whether to reach out now.\n"
        "Goals: build connection, show care, and be useful without being spammy.\n"
        "Rules:\n"
        "- Use memory/context and recent conversation themes.\n"
        "- If no meaningful outreach is appropriate now, return exactly __NO_NOTIFY__.\n"
        "- If outreach is appropriate, send one concise, natural message.\n"
        "- Prefer varied check-ins/questions/shares over repetitive phrasing.\n"
        "- If sharing content, pick one relevant thing only.\n"
        f"- Heartbeat cadence baseline: every {interval_hours} hour(s).\n"
    )


async def sync_proactive_heartbeat(
    *,
    runtime: Any,
    customer_id: str,
    directive_text: str,
) -> dict[str, Any]:
    cid = str(customer_id or "").strip()
    if not cid:
        return {"ok": False, "reason": "missing_customer_id"}

    routine_id = _proactive_heartbeat_routine_id(cid)
    wants_proactive = _directive_enables_proactive_mode(directive_text)
    default_hours = int(getattr(runtime, "_proactive_heartbeat_default_hours", 3))
    interval_hours = _extract_heartbeat_interval_hours(
        directive_text,
        default_hours=default_hours,
    )
    routine_name = "Proactive Heartbeat"

    if not wants_proactive:
        response = await runtime._request_with_backoff(
            "DELETE",
            f"/internal/scheduler/routine/{routine_id}",
            params={"customer_id": cid},
            timeout=8.0,
            retries=1,
        )
        if response.status_code != 200:
            return {
                "ok": False,
                "enabled": False,
                "routine_id": routine_id,
                "reason": f"heartbeat_disable_failed_http_{response.status_code}",
            }
        payload = response.json() if response.content else {}
        return {
            "ok": True,
            "enabled": False,
            "routine_id": routine_id,
            "removed": bool(payload.get("ok", False)),
            "interval_hours": interval_hours,
        }

    create = await runtime._request_with_backoff(
        "POST",
        "/internal/scheduler/routine",
        json_body={
            "id": routine_id,
            "name": routine_name,
            "schedule": f"0 */{interval_hours} * * *",
            "is_cron": True,
            "enabled": True,
            "payload": {
                "customer_id": cid,
                "notify_user": True,
                "proactive_heartbeat": True,
                "heartbeat_interval_hours": interval_hours,
                "message": _build_proactive_heartbeat_prompt(interval_hours),
            },
        },
        timeout=10.0,
        retries=1,
    )
    if create.status_code != 200:
        return {
            "ok": False,
            "enabled": True,
            "routine_id": routine_id,
            "interval_hours": interval_hours,
            "reason": f"heartbeat_enable_failed_http_{create.status_code}",
        }
    result = create.json() if create.content else {}
    return {
        "ok": True,
        "enabled": True,
        "routine_id": str(result.get("id", routine_id)).strip() or routine_id,
        "name": routine_name,
        "interval_hours": interval_hours,
        "schedule": f"0 */{interval_hours} * * *",
    }
