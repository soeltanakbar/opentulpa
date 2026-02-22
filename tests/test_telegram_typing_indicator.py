from __future__ import annotations

import asyncio

import pytest

from opentulpa.interfaces.telegram.relay import _emit_typing_until_done


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    async def send_chat_action(
        self,
        *,
        chat_id: int | str,
        action: str = "typing",
    ) -> bool:
        self.calls.append((int(chat_id), action))
        return True


@pytest.mark.asyncio
async def test_emit_typing_until_done_sends_typing_action() -> None:
    client = _FakeClient()
    stop = asyncio.Event()

    async def _stop_soon() -> None:
        await asyncio.sleep(0.01)
        stop.set()

    stopper = asyncio.create_task(_stop_soon())
    await _emit_typing_until_done(client=client, chat_id=42, stop_event=stop)  # type: ignore[arg-type]
    await stopper

    assert client.calls
    assert all(chat_id == 42 for chat_id, _ in client.calls)
    assert all(action == "typing" for _, action in client.calls)

