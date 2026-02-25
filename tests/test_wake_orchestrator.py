from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from opentulpa.application.wake_orchestrator import WakeOrchestrator


class _FakeContextEvents:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def add_event(self, **kwargs: Any) -> int:
        self.events.append(kwargs)
        return len(self.events)


class _FakeTelegramChat:
    def __init__(self) -> None:
        self.touched: list[int] = []

    async def relay_event(self, **_: Any) -> list[dict[str, Any]]:
        return [{"chat_id": 166, "text": "wake update"}]

    async def relay_task_event(self, **_: Any) -> list[dict[str, Any]]:
        return [{"chat_id": 166, "text": "task update"}]

    def touch_assistant_message(self, chat_id: int) -> None:
        self.touched.append(int(chat_id))


class _FakeTelegramClient:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_message(
        self,
        *,
        chat_id: int | str,
        text: str,
        parse_mode: str | None = None,
    ) -> bool:
        self.sent.append(
            {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }
        )
        return True


class _FakeApprovals:
    def __init__(self) -> None:
        self.flush_calls: list[dict[str, str]] = []

    async def flush_deferred_challenges(
        self,
        *,
        origin_interface: str,
        origin_conversation_id: str,
    ) -> int:
        self.flush_calls.append(
            {
                "origin_interface": origin_interface,
                "origin_conversation_id": origin_conversation_id,
            }
        )
        return 1


@pytest.mark.asyncio
async def test_routine_event_flushes_deferred_approval_challenges() -> None:
    settings = SimpleNamespace(telegram_bot_token="test-token")
    context_events = _FakeContextEvents()
    chat = _FakeTelegramChat()
    client = _FakeTelegramClient()
    approvals = _FakeApprovals()
    runtime = object()

    orchestrator = WakeOrchestrator(
        settings=settings,
        get_context_events=lambda: context_events,
        get_telegram_chat=lambda: chat,
        get_telegram_client=lambda: client,
        get_agent_runtime=lambda: runtime,
        get_approvals=lambda: approvals,
    )

    await orchestrator.handle_event(
        {
            "type": "routine_event",
            "event_type": "scheduled",
            "customer_id": "telegram_166",
            "routine_id": "rtn_123",
            "routine_name": "Test Routine",
            "notify_user": True,
            "payload": {
                "customer_id": "telegram_166",
                "notify_user": True,
                "message": "do the thing",
            },
        }
    )

    assert approvals.flush_calls == [
        {
            "origin_interface": "telegram",
            "origin_conversation_id": "166",
        }
    ]
    assert client.sent
