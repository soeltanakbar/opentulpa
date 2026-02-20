"""Telegram reply streaming and wake-event relays."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from opentulpa.agent.runtime import MergedInputSuppressedError
from opentulpa.core.ids import new_short_id
from opentulpa.interfaces.telegram.client import TelegramClient
from opentulpa.interfaces.telegram.constants import DEBUG_LOG_PATH, LOW_SIGNAL_REPLIES


def normalize_reply_text(text: str) -> str:
    import re

    t = text.strip().lower()
    t = re.sub(r"[.!?]+$", "", t)
    return " ".join(t.split())


def is_low_signal_reply(text: str) -> bool:
    normalized = normalize_reply_text(text)
    if not normalized:
        return True
    return normalized in LOW_SIGNAL_REPLIES


def debug_log(*, hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    try:
        payload = {
            "runId": "telegram",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


async def _push_loading_sequence(
    *,
    client: TelegramClient,
    chat_id: int,
    message_id: int | None = None,
    delay_seconds: float = 0.22,
    stop_event: asyncio.Event | None = None,
    previous_marker: str | None = None,
) -> tuple[int | None, str | None]:
    sequence = ("...", "..", ".", "..", "...")
    current_id = message_id
    last_marker = previous_marker
    for marker in sequence:
        if stop_event is not None and stop_event.is_set():
            break
        if marker == last_marker:
            continue
        current_id = (
            await client.upsert_stream_message(
                chat_id=chat_id,
                text=marker,
                message_id=current_id,
                parse_mode=None,
                allow_fallback_send=False,
            )
            or current_id
        )
        last_marker = marker
        if stop_event is None:
            await asyncio.sleep(delay_seconds)
            continue
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=delay_seconds)
    return current_id, last_marker


async def stream_langgraph_reply_to_telegram(
    *,
    agent_runtime: Any,
    thread_id: str,
    customer_id: str,
    text: str,
    bot_token: str,
    chat_id: int,
) -> tuple[str | None, bool]:
    stream_message_id: int | None = None
    last_streamed = ""
    final_reply = None
    client = TelegramClient(bot_token)
    loader_stop = asyncio.Event()
    stream_state: dict[str, int | None] = {"message_id": stream_message_id}
    loader_last_marker: dict[str, str | None] = {"value": None}

    async def _loader_loop() -> None:
        while not loader_stop.is_set():
            stream_state["message_id"], loader_last_marker["value"] = await _push_loading_sequence(
                client=client,
                chat_id=chat_id,
                message_id=stream_state.get("message_id"),
                stop_event=loader_stop,
                previous_marker=loader_last_marker["value"],
            )

    loader_task = asyncio.create_task(_loader_loop())
    suppressed = False
    first_token_timeout_s = 45.0
    stream_idle_timeout_s = 120.0
    try:
        stream = agent_runtime.astream_text(
            thread_id=thread_id,
            customer_id=customer_id,
            text=text,
        )
        stream_iter = stream.__aiter__()
        while True:
            timeout_s = first_token_timeout_s if not last_streamed else stream_idle_timeout_s
            try:
                partial = await asyncio.wait_for(stream_iter.__anext__(), timeout=timeout_s)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                with suppress(Exception):
                    await stream.aclose()
                if not loader_stop.is_set():
                    loader_stop.set()
                    with suppress(Exception):
                        await loader_task
                stream_message_id = stream_state.get("message_id")
                timeout_text = (
                    "Still working, but the model response timed out. "
                    "Please retry in a moment."
                )
                stream_message_id = (
                    await client.upsert_stream_message(
                        chat_id=chat_id,
                        text=timeout_text,
                        message_id=stream_message_id,
                        parse_mode="HTML",
                    )
                    or stream_message_id
                )
                stream_state["message_id"] = stream_message_id
                final_reply = timeout_text
                break
            if not isinstance(partial, str):
                continue
            current = partial.strip()
            if not current or is_low_signal_reply(current) or current == last_streamed:
                continue
            if not loader_stop.is_set():
                loader_stop.set()
                with suppress(Exception):
                    await loader_task
            stream_message_id = stream_state.get("message_id")
            stream_message_id = (
                await client.upsert_stream_message(
                    chat_id=chat_id,
                    text=current,
                    message_id=stream_message_id,
                    parse_mode="HTML",
                )
                or stream_message_id
            )
            last_streamed = current
            final_reply = current
            stream_state["message_id"] = stream_message_id
    except MergedInputSuppressedError:
        suppressed = True
    if not loader_stop.is_set():
        loader_stop.set()
    with suppress(Exception):
        await loader_task
    return final_reply, suppressed


async def relay_task_event_via_main_agent(
    *,
    customer_id: str,
    task_id: str,
    event_type: str,
    payload: dict[str, Any],
    state_store: Any,
    find_session_slots: Callable[[str], list[dict[str, Any]]],
    agent_runtime: Any | None = None,
) -> list[dict[str, Any]]:
    return await relay_event_via_main_agent(
        customer_id=customer_id,
        event_label=f"task/{event_type}",
        payload={
            "task_id": task_id,
            "event_type": event_type,
            "payload": payload,
        },
        state_store=state_store,
        find_session_slots=find_session_slots,
        agent_runtime=agent_runtime,
    )


async def relay_event_via_main_agent(
    *,
    customer_id: str,
    event_label: str,
    payload: dict[str, Any],
    state_store: Any,
    find_session_slots: Callable[[str], list[dict[str, Any]]],
    agent_runtime: Any | None = None,
) -> list[dict[str, Any]]:
    slots = find_session_slots(customer_id)
    if not slots:
        return []
    if agent_runtime is None:
        raise RuntimeError("Agent runtime unavailable for wake relay")

    instruction = (
        "System update: a background event occurred.\n"
        "Respond with concise plain-language status, what happened, and next action.\n"
        f"- event: {event_label}\n"
        f"- payload: {json.dumps(payload, ensure_ascii=False)[:4000]}"
    )
    replies: list[dict[str, Any]] = []
    for slot in slots:
        chat_id = int(slot["chat_id"])
        chat_key = str(chat_id)

        def _ensure_wake_thread_id(state: dict[str, Any], _chat_key: str = chat_key) -> str:
            sessions = state.get("sessions")
            if not isinstance(sessions, dict):
                sessions = {}
            raw_slot = sessions.get(_chat_key)
            if not isinstance(raw_slot, dict):
                raw_slot = {}
            wake_thread_id = str(raw_slot.get("wake_thread_id", "")).strip()
            if not wake_thread_id:
                wake_thread_id = new_short_id("wake")
                raw_slot["wake_thread_id"] = wake_thread_id
                sessions[_chat_key] = raw_slot
                state["sessions"] = sessions
            return wake_thread_id

        wake_thread_id = state_store.update(_ensure_wake_thread_id)
        try:
            text = await agent_runtime.ainvoke_text(
                thread_id=wake_thread_id,
                customer_id=customer_id,
                text=instruction,
                include_pending_context=False,
            )
            if text and text.strip():
                replies.append({"chat_id": chat_id, "text": text.strip()})
        except Exception:
            continue
    if not replies:
        raise RuntimeError("Main agent did not produce a wake reply")
    return replies
