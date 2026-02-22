from __future__ import annotations

import json

import pytest

from opentulpa.approvals.adapters.telegram import TelegramApprovalAdapter
from opentulpa.approvals.models import ApprovalRecord


class _FakeTelegramClient:
    def __init__(self) -> None:
        self.last_text: str | None = None
        self.last_markup: dict | None = None
        self.sent_count = 0

    async def send_message(self, *, chat_id, text, parse_mode="HTML", reply_markup=None):
        self.last_text = str(text)
        self.last_markup = reply_markup
        self.sent_count += 1
        return True


@pytest.mark.asyncio
async def test_approval_message_uses_full_browser_task_preview() -> None:
    fake = _FakeTelegramClient()
    adapter = TelegramApprovalAdapter(client=fake)  # type: ignore[arg-type]
    long_task = (
        "1. Go to https://www.moltbook.com.\n"
        "2. Attempt to 'Resend Verification' or 'Register' again using "
        "'opentulpa_kv@agentmail.to'.\n"
        "3. Once the new email is sent, wait for it and continue with verification.\n"
        "4. Capture final API key and profile URL.\n"
    )
    rec = ApprovalRecord(
        id="apr_test01",
        customer_id="telegram_1",
        thread_id="chat-1",
        origin_interface="telegram",
        origin_user_id="1",
        origin_conversation_id="1",
        action_name="browser_use_run",
        action_args_json=json.dumps({"task": long_task}),
        recipient_scope="external",
        impact_type="write",
        summary="run browser task: truncated summary",
        reason="policy_matrix",
        confidence=0.9,
        status="pending",
        created_at="2026-02-20T00:00:00+00:00",
        expires_at="2026-02-20T00:10:00+00:00",
        decided_at=None,
        executed_at=None,
        decision_actor_id=None,
    )

    ok = await adapter.send_challenge(rec)
    assert ok is True
    assert fake.last_text is not None
    assert "Attempt to 'Resend Verification' or 'Register' again" in fake.last_text
    assert "Once the new email is sent" in fake.last_text
    assert isinstance(fake.last_markup, dict)


@pytest.mark.asyncio
async def test_queue_then_flush_delivers_afterward() -> None:
    fake = _FakeTelegramClient()
    adapter = TelegramApprovalAdapter(client=fake)  # type: ignore[arg-type]
    rec = ApprovalRecord(
        id="apr_test02",
        customer_id="telegram_2",
        thread_id="chat-2",
        origin_interface="telegram",
        origin_user_id="2",
        origin_conversation_id="2",
        action_name="routine_create",
        action_args_json=json.dumps({"name": "Prep reminder"}),
        recipient_scope="unknown",
        impact_type="write",
        summary="execute routine_create",
        reason="policy_matrix",
        confidence=0.9,
        status="pending",
        created_at="2026-02-20T00:00:00+00:00",
        expires_at="2026-02-20T00:10:00+00:00",
        decided_at=None,
        executed_at=None,
        decision_actor_id=None,
    )

    queued = await adapter.queue_challenge(rec)
    assert queued is True
    assert fake.sent_count == 0

    delivered = await adapter.flush_challenges(chat_id="2")
    assert delivered == 1
    assert fake.sent_count == 1
    assert fake.last_text is not None
    assert "Approval needed for external-impact action." in fake.last_text
