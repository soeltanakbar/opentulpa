from __future__ import annotations

import asyncio

import pytest

from opentulpa.interfaces.telegram.relay import _push_loading_sequence


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str, int | None, str | None, bool]] = []

    async def upsert_stream_message(
        self,
        *,
        chat_id: int | str,
        text: str,
        message_id: int | None = None,
        parse_mode: str | None = None,
        allow_fallback_send: bool = True,
    ) -> int | None:
        cid = int(chat_id)
        self.calls.append((cid, text, message_id, parse_mode, allow_fallback_send))
        return 100 if message_id is None else message_id


@pytest.mark.asyncio
async def test_push_loading_sequence_cycles_markers() -> None:
    client = _FakeClient()
    message_id, last_marker = await _push_loading_sequence(
        client=client,  # type: ignore[arg-type]
        chat_id=42,
        message_id=None,
        delay_seconds=0.0,
    )

    assert message_id == 100
    assert last_marker == "..."
    assert [c[1] for c in client.calls] == ["...", "..", ".", "..", "..."]
    assert all(c[3] is None for c in client.calls)
    assert all(c[4] is False for c in client.calls)


@pytest.mark.asyncio
async def test_push_loading_sequence_stops_early_when_event_set() -> None:
    client = _FakeClient()
    stop = asyncio.Event()

    async def trigger_stop() -> None:
        await asyncio.sleep(0.01)
        stop.set()

    stopper = asyncio.create_task(trigger_stop())
    message_id, _ = await _push_loading_sequence(
        client=client,  # type: ignore[arg-type]
        chat_id=42,
        message_id=None,
        delay_seconds=0.1,
        stop_event=stop,
    )
    await stopper

    assert message_id == 100
    assert 1 <= len(client.calls) < 5
    assert client.calls[0][1] == "..."


@pytest.mark.asyncio
async def test_push_loading_sequence_skips_repeated_marker_between_cycles() -> None:
    client = _FakeClient()
    message_id, last = await _push_loading_sequence(
        client=client,  # type: ignore[arg-type]
        chat_id=42,
        message_id=None,
        delay_seconds=0.0,
    )
    message_id, last = await _push_loading_sequence(
        client=client,  # type: ignore[arg-type]
        chat_id=42,
        message_id=message_id,
        delay_seconds=0.0,
        previous_marker=last,
    )
    assert message_id == 100
    # second cycle should start from ".." because previous marker is "..."
    assert [c[1] for c in client.calls][5:9] == ["..", ".", "..", "..."]
