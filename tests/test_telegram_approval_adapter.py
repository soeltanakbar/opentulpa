from __future__ import annotations

import json

import pytest

from opentulpa.approvals.adapters.telegram import TelegramApprovalAdapter
from opentulpa.approvals.models import ApprovalRecord


class _FakeTelegramClient:
    def __init__(self) -> None:
        self.last_text: str | None = None
        self.last_markup: dict | None = None

    async def send_message(self, *, chat_id, text, parse_mode="HTML", reply_markup=None):
        self.last_text = str(text)
        self.last_markup = reply_markup
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
