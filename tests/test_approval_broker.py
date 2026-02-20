from __future__ import annotations

from pathlib import Path

import pytest

from opentulpa.approvals.broker import ApprovalBroker
from opentulpa.approvals.store import PendingApprovalStore


class _CaptureAdapter:
    name = "capture"
    interactive = True

    def __init__(self) -> None:
        self.sent: list[object] = []

    async def send_challenge(self, approval) -> bool:  # type: ignore[no-untyped-def]
        self.sent.append(approval)
        return True


class _ClassifierRuntime:
    async def classify_guardrail_intent(self, *, action_name: str, action_args: dict[str, object]) -> dict[str, object]:
        if action_name == "browser_use_run":
            task = str(action_args.get("task", "")).strip().lower()
            if "submit" in task or "buy" in task:
                return {
                    "ok": True,
                    "impact_type": "write",
                    "recipient_scope": "unknown",
                    "confidence": 0.9,
                    "reason": "task includes write intent",
                }
            return {
                "ok": True,
                "impact_type": "read",
                "recipient_scope": "unknown",
                "confidence": 0.9,
                "reason": "task is read-only",
            }
        if action_name == "tulpa_run_terminal":
            return {"ok": False, "error": "classifier_down"}
        return {
            "ok": True,
            "impact_type": "write",
            "recipient_scope": "unknown",
            "confidence": 0.7,
            "reason": "default",
        }


def _origin_resolver(customer_id: str, thread_id: str) -> dict[str, str]:
    assert customer_id
    assert thread_id
    return {
        "origin_interface": "telegram",
        "origin_user_id": "42",
        "origin_conversation_id": "4242",
    }


@pytest.mark.asyncio
async def test_self_target_send_is_not_gated(tmp_path: Path) -> None:
    broker = ApprovalBroker(
        store=PendingApprovalStore(db_path=tmp_path / "approvals.db"),
        runtime=_ClassifierRuntime(),
        adapters={"telegram": _CaptureAdapter()},
        origin_resolver=_origin_resolver,
    )
    result = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="uploaded_file_send",
        action_args={"file_id": "file_abc"},
    )
    assert result["gate"] == "allow"
    assert result.get("approval_id") is None


@pytest.mark.asyncio
async def test_external_action_requires_approval_and_reuses_pending(tmp_path: Path) -> None:
    adapter = _CaptureAdapter()
    broker = ApprovalBroker(
        store=PendingApprovalStore(db_path=tmp_path / "approvals.db"),
        runtime=_ClassifierRuntime(),
        adapters={"telegram": adapter},
        origin_resolver=_origin_resolver,
    )
    request = {
        "customer_id": "telegram_42",
        "thread_id": "chat-42",
        "action_name": "slack_post",
        "action_args": {"channel_id": "C123", "text": "hello"},
    }
    first = await broker.evaluate_action(**request)
    second = await broker.evaluate_action(**request)
    assert first["gate"] == "require_approval"
    assert second["gate"] == "require_approval"
    assert first["approval_id"] == second["approval_id"]
    assert len(adapter.sent) == 2


@pytest.mark.asyncio
async def test_browser_read_vs_write_intent(tmp_path: Path) -> None:
    broker = ApprovalBroker(
        store=PendingApprovalStore(db_path=tmp_path / "approvals.db"),
        runtime=_ClassifierRuntime(),
        adapters={"telegram": _CaptureAdapter()},
        origin_resolver=_origin_resolver,
    )
    read_result = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="browser_use_run",
        action_args={"task": "Open docs and summarize the page"},
    )
    write_result = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="browser_use_run",
        action_args={"task": "Open this site and submit the form"},
    )
    assert read_result["gate"] == "allow"
    assert write_result["gate"] == "require_approval"


@pytest.mark.asyncio
async def test_classifier_failure_defaults_to_approval_required(tmp_path: Path) -> None:
    broker = ApprovalBroker(
        store=PendingApprovalStore(db_path=tmp_path / "approvals.db"),
        runtime=_ClassifierRuntime(),
        adapters={"telegram": _CaptureAdapter()},
        origin_resolver=_origin_resolver,
    )
    result = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="tulpa_run_terminal",
        action_args={"command": "echo hi"},
    )
    assert result["gate"] == "require_approval"
    assert result["reason"] == "guardrail_uncertain"


@pytest.mark.asyncio
async def test_exact_duplicate_after_executed_is_denied_without_reprompt(tmp_path: Path) -> None:
    adapter = _CaptureAdapter()
    broker = ApprovalBroker(
        store=PendingApprovalStore(db_path=tmp_path / "approvals.db"),
        runtime=_ClassifierRuntime(),
        adapters={"telegram": adapter},
        origin_resolver=_origin_resolver,
    )
    request = {
        "customer_id": "telegram_42",
        "thread_id": "chat-42",
        "action_name": "slack_post",
        "action_args": {"channel_id": "C123", "text": "hello"},
    }

    first = await broker.evaluate_action(**request)
    approval_id = str(first.get("approval_id", "")).strip()
    assert first["gate"] == "require_approval"
    assert approval_id

    decided = await broker.decide(
        approval_id=approval_id,
        decision="approve",
        actor_interface="telegram",
        actor_id="42",
    )
    assert decided["ok"] is True
    assert decided["status"] == "approved"

    async def _executor(action_name: str, action_args: dict[str, object], customer_id: str) -> dict[str, object]:
        return {"ok": True, "action_name": action_name, "action_args": action_args, "customer_id": customer_id}

    executed = await broker.execute_approved_action(
        approval_id=approval_id,
        customer_id="telegram_42",
        executor=_executor,
    )
    assert executed["ok"] is True

    again = await broker.evaluate_action(**request)
    assert again["gate"] == "deny"
    assert again["reason"] == "already_executed_recent_duplicate"
    assert again.get("approval_id") is None
    assert len(adapter.sent) == 1


@pytest.mark.asyncio
async def test_browser_task_duplicate_summary_after_executed_is_denied(tmp_path: Path) -> None:
    adapter = _CaptureAdapter()
    broker = ApprovalBroker(
        store=PendingApprovalStore(db_path=tmp_path / "approvals.db"),
        runtime=_ClassifierRuntime(),
        adapters={"telegram": adapter},
        origin_resolver=_origin_resolver,
    )
    first = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="browser_use_run",
        action_args={
            "task": "Go to agentmail.to and submit registration",
            "max_steps": 18,
            "sessionId": "sess_a",
        },
    )
    approval_id = str(first.get("approval_id", "")).strip()
    assert first["gate"] == "require_approval"
    assert approval_id

    decided = await broker.decide(
        approval_id=approval_id,
        decision="approve",
        actor_interface="telegram",
        actor_id="42",
    )
    assert decided["ok"] is True

    async def _executor(action_name: str, action_args: dict[str, object], customer_id: str) -> dict[str, object]:
        return {"ok": True}

    executed = await broker.execute_approved_action(
        approval_id=approval_id,
        customer_id="telegram_42",
        executor=_executor,
    )
    assert executed["ok"] is True

    # Same task text, drifted transient args: should not reprompt immediately.
    again = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="browser_use_run",
        action_args={
            "task": "Go to agentmail.to and submit registration",
            "max_steps": 25,
            "sessionId": "sess_b",
        },
    )
    assert again["gate"] == "deny"
    assert again["reason"] == "already_executed_recent_browser_task"
    assert again.get("approval_id") is None
