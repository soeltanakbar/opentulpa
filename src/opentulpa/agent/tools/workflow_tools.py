"""Workflow/scheduler/task LangChain tool bundle."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from langchain.tools import tool

from opentulpa.agent.result_models import ToolGuardrailDecision
from opentulpa.policy.execution_boundary import ExecutionBoundaryContext


def build_workflow_tools(
    *,
    runtime: Any,
    boundary_guard: Any,
    normalize_execution_origin: Callable[..., str],
    approval_pending_payload: Callable[..., dict[str, Any]],
    normalize_cleanup_paths: Callable[[list[str] | None], list[str]],
    looks_like_shell_command: Callable[[str], bool],
) -> dict[str, Any]:
    """Build task and scheduler workflow tools."""

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
        implementation_command: str,
        customer_id: str,
        notify_user: bool = True,
        cleanup_paths: list[str] | None = None,
        thread_id: str = "",
        execution_origin: str | None = None,
        preapproved: bool = False,
        guard_context: dict[str, Any] | None = None,
    ) -> Any:
        """
        Create a scheduled routine.
        - Recurring: cron (e.g. "0 9 * * *")
        - One-time: local ISO datetime (e.g. "2026-02-18T23:45:00+08:00")
        - implementation_command: planned shell/script command used for guardrail evaluation.
        - cleanup_paths: optional repo-relative file paths to remove when deleting this automation.
        """
        safe_name = str(name or "").strip()
        safe_schedule = str(schedule or "").strip()
        safe_message = str(message or "").strip()
        safe_command = str(implementation_command or "").strip()
        safe_customer = str(customer_id or "").strip()
        if not safe_name:
            return {"error": "routine_create failed: name is required"}
        if not safe_schedule:
            return {"error": "routine_create failed: schedule is required"}
        if not safe_customer:
            return {"error": "routine_create failed: customer_id is required"}
        if not safe_command:
            return {
                "error": (
                    "ROUTINE_IMPLEMENTATION_COMMAND_REQUIRED: routine_create requires "
                    "implementation_command (concrete shell/script command)."
                )
            }
        if not looks_like_shell_command(safe_command):
            return {
                "error": (
                    "ROUTINE_IMPLEMENTATION_COMMAND_INVALID: implementation_command must be a "
                    "concrete shell command (executable + args)."
                )
            }

        normalized_origin = normalize_execution_origin(
            thread_id=thread_id,
            execution_origin=execution_origin,
        )

        guard_payload = guard_context if isinstance(guard_context, dict) else {}
        previous_user = str(guard_payload.get("previous_user_message", "")).strip()
        previous_assistant = str(guard_payload.get("previous_assistant_message", "")).strip()
        decision = await boundary_guard.evaluate(
            ExecutionBoundaryContext(
                customer_id=safe_customer,
                thread_id=str(thread_id or "").strip() or f"chat-{safe_customer}",
                action_name="routine_create",
                action_args={
                    "name": safe_name,
                    "schedule": safe_schedule,
                    "message": safe_message[:1200],
                    "customer_id": safe_customer,
                    "notify_user": bool(notify_user),
                    "implementation_command": safe_command,
                },
                execution_origin=normalized_origin,
                preapproved=bool(preapproved),
                action_note=(
                    "Routine creation with planned implementation command. "
                    "Classify external write side effects for future scheduled behavior. "
                    f"previous_user_message={previous_user[:800]} "
                    f"previous_assistant_message={previous_assistant[:800]}"
                ),
            )
        )
        guard_decision = ToolGuardrailDecision.from_any(
            decision,
            default_summary="execute routine_create",
            default_reason="guardrail_invalid_payload",
        )
        gate = str(guard_decision.gate).strip().lower()
        if gate == "require_approval":
            return approval_pending_payload(
                action_name="routine_create",
                command_preview=safe_command,
                decision=guard_decision,
            )
        if gate == "deny":
            return {
                "ok": False,
                "status": "denied",
                "gate": "deny",
                "reason": str(guard_decision.reason or "guardrail_denied").strip(),
            }

        auto_notify = bool(notify_user)
        safe_cleanup_paths = normalize_cleanup_paths(cleanup_paths)

        r = await runtime._request_with_backoff(
            "POST",
            "/internal/scheduler/routine",
            json_body={
                "name": safe_name,
                "schedule": safe_schedule,
                "payload": {
                    "message": safe_message,
                    "customer_id": safe_customer,
                    "notify_user": auto_notify,
                    "notification_opt_out": not auto_notify,
                    "cleanup_paths": safe_cleanup_paths,
                },
                "is_cron": " " in safe_schedule and len(safe_schedule.split()) >= 5,
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
                "cleanup_paths": normalize_cleanup_paths(cleanup_paths),
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
