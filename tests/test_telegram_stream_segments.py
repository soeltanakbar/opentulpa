from __future__ import annotations

import asyncio

import pytest

from opentulpa.agent.runtime import STREAM_WAIT_SIGNAL
from opentulpa.interfaces.telegram import relay as relay_module


class _SegmentedRuntime:
    async def astream_text(self, **kwargs):
        yield "I have access to your inbox. I will check it now."
        yield STREAM_WAIT_SIGNAL
        await asyncio.sleep(0.02)
        yield "I checked your inbox. 3 priority emails found."


class _FakeTelegramClient:
    def __init__(self, bot_token: str) -> None:
        self.bot_token = bot_token
        self.calls: list[tuple[int | str, str, int | None, str | None]] = []
        self._next_id = 100

    async def upsert_stream_message(
        self,
        *,
        chat_id: int | str,
        text: str,
        message_id: int | None = None,
        parse_mode: str | None = None,
        allow_fallback_send: bool = True,
        reply_markup=None,
    ) -> int | None:
        self.calls.append((chat_id, text, message_id, parse_mode))
        if message_id is None:
            self._next_id += 1
            return self._next_id
        return message_id


@pytest.mark.asyncio
async def test_stream_creates_new_message_for_new_meaningful_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=_SegmentedRuntime(),
        thread_id="chat-1",
        customer_id="telegram_1",
        text="check inbox",
        bot_token="dummy",
        chat_id=1,
    )

    assert suppressed is False
    assert "priority emails" in str(final or "").lower()

    meaningful = [
        call
        for call in fake_client.calls
        if call[1] not in {"...", "..", "."}
    ]
    assert len(meaningful) >= 2
    # meaningful segments should not be forced into a single edited message
    first_msg_id = meaningful[0][2]
    second_msg_id = meaningful[-1][2]
    assert first_msg_id != second_msg_id
