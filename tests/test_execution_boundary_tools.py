from __future__ import annotations

from typing import Any

import pytest

from opentulpa.agent.tools_registry import register_runtime_tools


class _Response:
    def __init__(self, status_code: int, payload: dict[str, Any] | list[Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = "" if payload is None else str(payload)
        self.content = b"" if payload is None else b"x"

    def json(self) -> dict[str, Any] | list[Any]:
        return self._payload if self._payload is not None else {}


class _DummyRuntime:
    def __init__(
        self,
        responses: list[_Response],
        *,
        guard_result: dict[str, Any] | None = None,
    ) -> None:
        self._responses = list(responses)
        self._guard_result = guard_result or {"gate": "allow", "reason": "ok", "summary": "execute"}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.guard_calls: list[dict[str, Any]] = []

    async def _request_with_backoff(self, method: str, path: str, **kwargs: Any) -> _Response:
        self.calls.append((method, path, kwargs))
        if not self._responses:
            raise RuntimeError("unexpected internal API call")
        return self._responses.pop(0)

    async def evaluate_tool_guardrail(
        self,
        *,
        customer_id: str,
        thread_id: str,
        action_name: str,
        action_args: dict[str, Any],
        action_note: str | None = None,
    ) -> dict[str, Any]:
        self.guard_calls.append(
            {
                "customer_id": customer_id,
                "thread_id": thread_id,
                "action_name": action_name,
                "action_args": action_args,
                "action_note": action_note,
            }
        )
        return dict(self._guard_result)


@pytest.mark.asyncio
async def test_terminal_interactive_requires_approval_before_execution() -> None:
    runtime = _DummyRuntime(
        [],
        guard_result={
            "gate": "require_approval",
            "approval_id": "apr_boundary_1",
            "summary": "run terminal command",
            "reason": "external write side effect",
        },
    )
    tools = register_runtime_tools(runtime)
    result = await tools["tulpa_run_terminal"].ainvoke(
        {
            "command": "python3 tulpa_stuff/scripts/post_update.py",
            "working_dir": "tulpa_stuff",
            "customer_id": "telegram_1",
            "thread_id": "chat-1",
            "execution_origin": "interactive",
        }
    )
    assert result["status"] == "approval_pending"
    assert result["approval_id"] == "apr_boundary_1"
    assert len(runtime.guard_calls) == 1
    assert runtime.calls == []


@pytest.mark.asyncio
async def test_terminal_scheduled_execution_skips_guardrail() -> None:
    runtime = _DummyRuntime(
        [_Response(200, {"ok": True, "stdout": "done", "stderr": "", "returncode": 0})],
        guard_result={"gate": "require_approval", "approval_id": "apr_should_not_happen"},
    )
    tools = register_runtime_tools(runtime)
    result = await tools["tulpa_run_terminal"].ainvoke(
        {
            "command": "python3 tulpa_stuff/scripts/digest.py",
            "working_dir": "tulpa_stuff",
            "customer_id": "telegram_1",
            "thread_id": "wake_abc123",
            "execution_origin": "scheduled",
        }
    )
    assert result["ok"] is True
    assert result["execution_origin"] == "scheduled"
    assert len(runtime.guard_calls) == 0
    assert len(runtime.calls) == 1
    assert runtime.calls[0][1] == "/internal/tulpa/run_terminal"


@pytest.mark.asyncio
async def test_routine_create_saves_schedule_without_execution_artifact_metadata() -> None:
    runtime = _DummyRuntime(
        [_Response(200, {"ok": True, "id": "rtn_1"})],
        guard_result={"gate": "allow", "reason": "internal plan", "summary": "create routine"},
    )
    tools = register_runtime_tools(runtime)
    result = await tools["routine_create"].ainvoke(
        {
            "name": "Daily Digest",
            "schedule": "0 9 * * *",
            "message": "Prepare and send digest",
            "implementation_command": "python3 tulpa_stuff/scripts/digest.py",
            "implementation_working_dir": "tulpa_stuff",
            "implementation_timeout_seconds": 120,
            "customer_id": "telegram_1",
            "notify_user": True,
            "thread_id": "chat-1",
            "execution_origin": "interactive",
        }
    )
    assert result["ok"] is True
    assert result["id"] == "rtn_1"
    assert len(runtime.guard_calls) == 1
    assert len(runtime.calls) == 1
    sent = runtime.calls[0][2]["json_body"]
    assert "execution" not in sent["payload"]


@pytest.mark.asyncio
async def test_routine_create_pending_approval_does_not_save_schedule() -> None:
    runtime = _DummyRuntime(
        [],
        guard_result={
            "gate": "require_approval",
            "approval_id": "apr_routine_1",
            "summary": "create external write routine",
            "reason": "external write side effect",
        },
    )
    tools = register_runtime_tools(runtime)
    result = await tools["routine_create"].ainvoke(
        {
            "name": "Auto Post",
            "schedule": "0 */2 * * *",
            "message": "Post market reflections",
            "implementation_command": "python3 tulpa_stuff/scripts/post_agentx.py",
            "customer_id": "telegram_1",
            "thread_id": "chat-1",
            "execution_origin": "interactive",
        }
    )
    assert result["status"] == "approval_pending"
    assert result["approval_id"] == "apr_routine_1"
    assert len(runtime.guard_calls) == 1
    assert runtime.calls == []


@pytest.mark.asyncio
async def test_routine_create_requires_non_empty_implementation_command() -> None:
    runtime = _DummyRuntime([_Response(200, {"ok": True, "id": "rtn_unexpected"})])
    tools = register_runtime_tools(runtime)
    result = await tools["routine_create"].ainvoke(
        {
            "name": "Auto Post",
            "schedule": "0 */2 * * *",
            "message": "Post market reflections",
            "implementation_command": "   ",
            "customer_id": "telegram_1",
        }
    )
    assert str(result.get("error", "")).startswith("ROUTINE_IMPLEMENTATION_COMMAND_REQUIRED")
    assert runtime.calls == []
    assert runtime.guard_calls == []
