from __future__ import annotations

import pytest

from opentulpa.api.routes.telegram_webhook import _execute_approved_action_and_summarize


class _FakeContextEvents:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def add_event(self, **kwargs):  # type: ignore[no-untyped-def]
        self.events.append(kwargs)
        return len(self.events)


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def execute_tool(self, *, action_name: str, action_args: dict[str, object]):
        self.calls.append({"kind": "execute", "action_name": action_name, "action_args": action_args})
        return {
            "ok": True,
            "approval_id": "apr_test",
            "status": "executed",
            "action_name": "tulpa_write_file",
            "result": {"error": 'write failed: {"detail":"router missing"}'},
            "error": 'write failed: {"detail":"router missing"}',
        }

    async def ainvoke_text(  # type: ignore[no-untyped-def]
        self,
        *,
        thread_id: str,
        customer_id: str,
        text: str,
        include_pending_context: bool = True,
        recursion_limit_override: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "kind": "ainvoke",
                "thread_id": thread_id,
                "customer_id": customer_id,
                "include_pending_context": include_pending_context,
                "recursion_limit_override": recursion_limit_override,
                "text": text,
            }
        )
        return "I fixed the script path and re-ran setup. Here is your auth link."


@pytest.mark.asyncio
async def test_failed_approved_action_uses_autonomous_recovery_message() -> None:
    runtime = _FakeRuntime()
    context_events = _FakeContextEvents()
    decision_payload = {
        "customer_id": "telegram_42",
        "thread_id": "chat-42",
        "action_name": "tulpa_write_file",
        "summary": "write gmail_setup.py",
        "action_args": {"path": "tulpa_stuff/gmail_setup.py", "content": "print('x')"},
    }

    out = await _execute_approved_action_and_summarize(
        get_agent_runtime=lambda: runtime,
        get_context_events=lambda: context_events,
        approval_id="apr_test",
        decision_payload=decision_payload,
        chat_id=42,
    )

    assert "auth link" in out.lower()
    ainvoke_calls = [c for c in runtime.calls if c.get("kind") == "ainvoke"]
    assert len(ainvoke_calls) == 1
    assert ainvoke_calls[0].get("recursion_limit_override") == 48
