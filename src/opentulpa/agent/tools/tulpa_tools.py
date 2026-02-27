"""Tulpa file/terminal LangChain tool bundle."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain.tools import tool

from opentulpa.agent.result_models import ToolGuardrailDecision
from opentulpa.policy.execution_boundary import ExecutionBoundaryContext


def build_tulpa_tools(
    *,
    runtime: Any,
    boundary_guard: Any,
    normalize_execution_origin: Callable[..., str],
    approval_pending_payload: Callable[..., dict[str, Any]],
    looks_like_shell_command: Callable[[str], bool],
) -> dict[str, Any]:
    """Build tulpa file and terminal execution tools."""

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
        customer_id: str = "",
        thread_id: str = "",
        execution_origin: str | None = None,
        preapproved: bool = False,
        guard_context: dict[str, Any] | None = None,
    ) -> Any:
        """Run executable shell/script command through execution-boundary guard."""
        safe_command = str(command or "").strip()
        if not looks_like_shell_command(safe_command):
            return {
                "error": (
                    "Command rejected: provide a concrete shell command (executable + args), "
                    "not natural language."
                )
            }
        safe_timeout = max(5, min(int(timeout_seconds), 600))
        safe_customer = str(customer_id or "").strip()
        safe_thread = str(thread_id or "").strip()
        normalized_origin = normalize_execution_origin(
            thread_id=safe_thread,
            execution_origin=execution_origin,
        )

        guard_payload = guard_context if isinstance(guard_context, dict) else {}
        previous_user = str(guard_payload.get("previous_user_message", "")).strip()
        previous_assistant = str(guard_payload.get("previous_assistant_message", "")).strip()
        decision = await boundary_guard.evaluate(
            ExecutionBoundaryContext(
                customer_id=safe_customer,
                thread_id=safe_thread or (f"chat-{safe_customer}" if safe_customer else "interactive"),
                action_name="tulpa_run_terminal",
                action_args={
                    "command": safe_command,
                    "working_dir": str(working_dir or "").strip() or "tulpa_stuff",
                    "timeout_seconds": safe_timeout,
                    "execution_origin": normalized_origin,
                },
                execution_origin=normalized_origin,
                preapproved=bool(preapproved),
                action_note=(
                    "Execution-boundary guard check for terminal/script action. "
                    "Decide based on full command external write side effects. "
                    f"previous_user_message={previous_user[:800]} "
                    f"previous_assistant_message={previous_assistant[:800]}"
                ),
            )
        )
        guard_decision = ToolGuardrailDecision.from_any(
            decision,
            default_summary="execute tulpa_run_terminal",
            default_reason="guardrail_invalid_payload",
        )
        gate = str(guard_decision.gate).strip().lower()
        if gate == "require_approval":
            return approval_pending_payload(
                action_name="tulpa_run_terminal",
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

        r = await runtime._request_with_backoff(
            "POST",
            "/internal/tulpa/run_terminal",
            json_body={
                "command": safe_command,
                "working_dir": working_dir,
                "timeout_seconds": safe_timeout,
            },
            timeout=max(10.0, float(safe_timeout) + 10.0),
            retries=1,
        )
        if r.status_code != 200:
            return {"error": f"terminal failed: {r.text}"}
        payload = r.json()
        if isinstance(payload, dict):
            payload["execution_origin"] = normalized_origin
        return payload

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

    return {
        "tulpa_write_file": tulpa_write_file,
        "tulpa_validate_file": tulpa_validate_file,
        "tulpa_run_terminal": tulpa_run_terminal,
        "tulpa_read_file": tulpa_read_file,
        "tulpa_catalog": tulpa_catalog,
    }
