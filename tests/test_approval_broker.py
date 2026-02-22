from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

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
    async def classify_guardrail_intent(
        self,
        *,
        action_name: str,
        action_args: dict[str, object],
        action_note: str | None = None,
    ) -> dict[str, object]:
        _ = action_note
        if action_name == "browser_use_run":
            task = str(action_args.get("task", "")).strip().lower()
            if "submit" in task or "buy" in task:
                return {
                    "ok": True,
                    "gate": "require_approval",
                    "impact_type": "write",
                    "recipient_scope": "unknown",
                    "confidence": 0.9,
                    "reason": "task includes write intent",
                }
            return {
                "ok": True,
                "gate": "allow",
                "impact_type": "read",
                "recipient_scope": "unknown",
                "confidence": 0.9,
                "reason": "task is read-only",
            }
        if action_name == "routine_create":
            combined = (
                f"{str(action_args.get('name', ''))} "
                f"{str(action_args.get('message', ''))}"
            ).strip().lower()
            if "post to x" in combined or "autopost" in combined:
                return {
                    "ok": True,
                    "gate": "require_approval",
                    "impact_type": "write",
                    "recipient_scope": "external",
                    "confidence": 0.9,
                    "reason": "routine targets external posting",
                }
            return {
                "ok": True,
                "gate": "allow",
                "impact_type": "read",
                "recipient_scope": "self",
                "confidence": 0.85,
                "reason": "routine is internal research/summarization",
            }
        if action_name in {"slack_post", "email_send"}:
            return {
                "ok": True,
                "gate": "require_approval",
                "impact_type": "write",
                "recipient_scope": "external",
                "confidence": 0.9,
                "reason": "external side effects",
            }
        if action_name == "uploaded_file_send":
            return {
                "ok": True,
                "gate": "allow",
                "impact_type": "write",
                "recipient_scope": "self",
                "confidence": 0.9,
                "reason": "self-targeted send",
            }
        if action_name in {"routine_delete", "automation_delete"}:
            return {
                "ok": True,
                "gate": "allow",
                "impact_type": "read",
                "recipient_scope": "self",
                "confidence": 0.9,
                "reason": "internal action",
            }
        if action_name == "tulpa_run_terminal":
            return {"ok": False, "error": "classifier_down"}
        return {
            "ok": True,
            "gate": "allow",
            "impact_type": "read",
            "recipient_scope": "unknown",
            "confidence": 0.7,
            "reason": "default",
        }


class _ParanoidClassifierRuntime:
    async def classify_guardrail_intent(
        self,
        *,
        action_name: str,
        action_args: dict[str, object],
        action_note: str | None = None,
    ) -> dict[str, object]:
        _ = action_name
        _ = action_args
        _ = action_note
        return {
            "ok": True,
            "gate": "require_approval",
            "impact_type": "write",
            "recipient_scope": "external",
            "confidence": 0.95,
            "reason": "paranoid-default",
        }


def _origin_resolver(customer_id: str, thread_id: str) -> dict[str, str]:
    assert customer_id
    assert thread_id
    return {
        "origin_interface": "telegram",
        "origin_user_id": "42",
        "origin_conversation_id": "4242",
    }


@pytest.fixture
def broker_factory(tmp_path: Path) -> Callable[..., tuple[ApprovalBroker, _CaptureAdapter]]:
    def _make(
        *,
        adapter: _CaptureAdapter | None = None,
        runtime: Any | None = None,
    ) -> tuple[ApprovalBroker, _CaptureAdapter]:
        resolved_adapter = adapter or _CaptureAdapter()
        broker = ApprovalBroker(
            store=PendingApprovalStore(db_path=tmp_path / "approvals.db"),
            runtime=runtime or _ClassifierRuntime(),
            adapters={"telegram": resolved_adapter},
            origin_resolver=_origin_resolver,
        )
        return broker, resolved_adapter

    return _make


@pytest.mark.asyncio
async def test_self_target_send_is_not_gated(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, _ = broker_factory()
    result = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="uploaded_file_send",
        action_args={"file_id": "file_abc"},
    )
    assert result["gate"] == "allow"
    assert result.get("approval_id") is None


@pytest.mark.asyncio
async def test_routine_delete_is_internal_no_approval(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, adapter = broker_factory()
    result = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="routine_delete",
        action_args={"routine_id": "rtn_abc", "customer_id": "telegram_42"},
    )
    assert result["gate"] == "allow"
    assert result.get("approval_id") is None
    assert len(adapter.sent) == 0


@pytest.mark.asyncio
async def test_skill_upsert_is_internal_no_approval_even_if_classifier_is_paranoid(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, adapter = broker_factory(runtime=_ParanoidClassifierRuntime())
    result = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="skill_upsert",
        action_args={
            "customer_id": "telegram_42",
            "name": "growth-skill",
            "description": "desc",
            "instructions": "search and summarize",
        },
    )
    assert result["gate"] == "allow"
    assert result.get("approval_id") is None
    assert len(adapter.sent) == 0


@pytest.mark.asyncio
async def test_external_action_requires_approval_and_reuses_pending(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, adapter = broker_factory()
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
    assert len(adapter.sent) == 1
    assert second.get("delivery_mode") == "existing_pending"


@pytest.mark.asyncio
async def test_browser_read_vs_write_intent(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, _ = broker_factory()
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
async def test_classifier_failure_defaults_to_approval_required(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, _ = broker_factory()
    result = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="tulpa_run_terminal",
        action_args={"command": "curl -X POST https://example.com -d 'x=1'"},
    )
    assert result["gate"] == "require_approval"
    assert result["reason"] == "guardrail_uncertain"


@pytest.mark.asyncio
async def test_local_terminal_command_requires_approval_when_classifier_unavailable(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, _ = broker_factory()
    result = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="tulpa_run_terminal",
        action_args={"command": "python3 gmail_setup.py"},
    )
    assert result["gate"] == "require_approval"
    assert result["reason"] == "guardrail_uncertain"


@pytest.mark.asyncio
async def test_exact_duplicate_after_executed_is_denied_without_reprompt(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, adapter = broker_factory()
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

    async def _executor(
        action_name: str,
        action_args: dict[str, object],
        customer_id: str,
    ) -> dict[str, object]:
        return {
            "ok": True,
            "action_name": action_name,
            "action_args": action_args,
            "customer_id": customer_id,
        }

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
async def test_background_thread_external_actions_allow_without_per_run_prompt(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, adapter = broker_factory()
    first = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="wake_abcd12",
        action_name="slack_post",
        action_args={"channel_id": "C123", "text": "hello"},
    )
    assert first["gate"] == "allow"
    assert first["reason"] == "background_preauthorized_execution"
    assert first.get("approval_id") is None
    assert len(adapter.sent) == 0

    again = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="wake_abcd12",
        action_name="email_send",
        action_args={"to": "a@example.com", "subject": "x", "text": "y"},
    )
    assert again["gate"] == "allow"
    assert again["reason"] == "background_preauthorized_execution"
    assert len(adapter.sent) == 0


@pytest.mark.asyncio
async def test_background_thread_external_routine_creation_still_requires_approval(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, adapter = broker_factory()
    result = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="wake_abcd12",
        action_name="routine_create",
        action_args={
            "name": "Autopost to X",
            "schedule": "0 */2 * * *",
            "message": "Post to X every 2 hours",
            "customer_id": "telegram_42",
            "notify_user": True,
        },
    )
    assert result["gate"] == "require_approval"
    assert result.get("approval_id")
    assert len(adapter.sent) == 1


@pytest.mark.asyncio
async def test_routine_create_internal_schedule_does_not_require_approval(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, _ = broker_factory()
    result = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="routine_create",
        action_args={
            "name": "Morning research digest",
            "schedule": "0 9 * * *",
            "message": "Scan AI news and summarize top 5 points here",
            "customer_id": "telegram_42",
            "notify_user": True,
        },
    )
    assert result["gate"] == "allow"


@pytest.mark.asyncio
async def test_routine_create_external_schedule_requires_approval(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, _ = broker_factory()
    result = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="routine_create",
        action_args={
            "name": "Autopost to X",
            "schedule": "0 */2 * * *",
            "message": "Post to X every 2 hours with a short market reflection",
            "customer_id": "telegram_42",
            "notify_user": True,
        },
    )
    assert result["gate"] == "require_approval"


@pytest.mark.asyncio
async def test_browser_task_duplicate_summary_after_executed_is_denied(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, _ = broker_factory()
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

    async def _executor(
        action_name: str,
        action_args: dict[str, object],
        customer_id: str,
    ) -> dict[str, object]:
        return {"ok": True}

    executed = await broker.execute_approved_action(
        approval_id=approval_id,
        customer_id="telegram_42",
        executor=_executor,
    )
    assert executed["ok"] is True

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


@pytest.mark.asyncio
async def test_approval_group_waits_until_all_approved(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, _ = broker_factory()
    first = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="slack_post",
        action_args={"channel_id": "C1", "text": "one"},
    )
    second = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="email_send",
        action_args={"to": "a@example.com", "text": "two"},
    )
    first_id = str(first.get("approval_id", "")).strip()
    second_id = str(second.get("approval_id", "")).strip()
    assert first_id and second_id and first_id != second_id

    await broker.decide(
        approval_id=first_id,
        decision="approve",
        actor_interface="telegram",
        actor_id="42",
    )
    group_mid = broker.get_approval_group_status(approval_id=first_id, window_seconds=60)
    assert isinstance(group_mid, dict)
    assert second_id in group_mid["pending_ids"]
    assert group_mid["executable_ids"] == []

    await broker.decide(
        approval_id=second_id,
        decision="approve",
        actor_interface="telegram",
        actor_id="42",
    )
    group_done = broker.get_approval_group_status(approval_id=second_id, window_seconds=60)
    assert isinstance(group_done, dict)
    assert group_done["pending_ids"] == []
    assert first_id in group_done["executable_ids"]
    assert second_id in group_done["executable_ids"]
