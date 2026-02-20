"""Telegram reply streaming and wake-event relays."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from opentulpa.agent.runtime import MergedInputSuppressedError, STREAM_WAIT_SIGNAL
from opentulpa.core.ids import new_short_id
from opentulpa.interfaces.telegram.client import TelegramClient
from opentulpa.interfaces.telegram.constants import DEBUG_LOG_PATH, LOW_SIGNAL_REPLIES

logger = logging.getLogger(__name__)


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
    waiting_for_segment = True
    stream_state: dict[str, int | None] = {"message_id": stream_message_id}
    loader_last_marker: dict[str, str | None] = {"value": None}
    loader_stop: asyncio.Event | None = None
    loader_task: asyncio.Task[None] | None = None

    async def _start_loader(*, new_message: bool) -> None:
        nonlocal loader_stop, loader_task
        if loader_task is not None and not loader_task.done():
            return
        if new_message:
            stream_state["message_id"] = None
            loader_last_marker["value"] = None
        loader_stop = asyncio.Event()

        async def _loader_loop(local_stop: asyncio.Event) -> None:
            while not local_stop.is_set():
                stream_state["message_id"], loader_last_marker["value"] = await _push_loading_sequence(
                    client=client,
                    chat_id=chat_id,
                    message_id=stream_state.get("message_id"),
                    stop_event=local_stop,
                    previous_marker=loader_last_marker["value"],
                )

        loader_task = asyncio.create_task(_loader_loop(loader_stop))

    async def _stop_loader() -> None:
        nonlocal loader_stop, loader_task
        if loader_stop is not None and not loader_stop.is_set():
            loader_stop.set()
        if loader_task is not None:
            with suppress(Exception):
                await loader_task
        loader_stop = None
        loader_task = None

    await _start_loader(new_message=False)
    suppressed = False
    first_token_timeout_s = 90.0
    first_token_retry_timeout_s = 180.0
    stream_idle_timeout_s = 180.0
    stream_idle_retry_timeout_s = 240.0
    consecutive_timeouts = 0
    max_consecutive_timeouts = 2
    next_chunk_task: asyncio.Task[Any] | None = None
    logger.info(
        "telegram.stream start chat_id=%s thread_id=%s customer_id=%s text_chars=%s",
        chat_id,
        thread_id,
        customer_id,
        len(str(text or "")),
    )
    try:
        stream = agent_runtime.astream_text(
            thread_id=thread_id,
            customer_id=customer_id,
            text=text,
        )
        stream_iter = stream.__aiter__()
        while True:
            if not last_streamed:
                timeout_s = (
                    first_token_timeout_s
                    if consecutive_timeouts == 0
                    else first_token_retry_timeout_s
                )
            else:
                timeout_s = (
                    stream_idle_timeout_s
                    if consecutive_timeouts == 0
                    else stream_idle_retry_timeout_s
                )
            try:
                if next_chunk_task is None:
                    next_chunk_task = asyncio.create_task(stream_iter.__anext__())
                partial = await asyncio.wait_for(
                    asyncio.shield(next_chunk_task),
                    timeout=timeout_s,
                )
                next_chunk_task = None
            except StopAsyncIteration:
                next_chunk_task = None
                break
            except asyncio.TimeoutError:
                consecutive_timeouts += 1
                if consecutive_timeouts < max_consecutive_timeouts:
                    # Auto-retry once on transient model/provider stalls before failing user-visible.
                    logger.warning(
                        "telegram.stream timeout_retry chat_id=%s thread_id=%s customer_id=%s stage=%s",
                        chat_id,
                        thread_id,
                        customer_id,
                        "first_token" if not last_streamed else "idle",
                    )
                    continue
                if next_chunk_task is not None and not next_chunk_task.done():
                    next_chunk_task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        async with asyncio.timeout(1.0):
                            await next_chunk_task
                next_chunk_task = None
                with suppress(Exception):
                    async with asyncio.timeout(1.0):
                        await stream.aclose()
                await _stop_loader()
                stream_message_id = stream_state.get("message_id")
                timeout_text = (
                    "Still working, but the model response timed out. "
                    "Please retry in a moment."
                )
                logger.error(
                    "telegram.stream timeout_fail chat_id=%s thread_id=%s customer_id=%s stage=%s",
                    chat_id,
                    thread_id,
                    customer_id,
                    "first_token" if not last_streamed else "idle",
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
            if isinstance(partial, str) and partial == STREAM_WAIT_SIGNAL:
                if not waiting_for_segment:
                    waiting_for_segment = True
                    last_streamed = ""
                    await _stop_loader()
                    await _start_loader(new_message=True)
                continue
            if not isinstance(partial, str):
                continue
            consecutive_timeouts = 0
            current = partial.strip()
            if not current or is_low_signal_reply(current) or current == last_streamed:
                continue
            # Defensive boundary handling for streams that reset partial text without explicit signal.
            if last_streamed and not current.startswith(last_streamed):
                waiting_for_segment = True
                last_streamed = ""
                await _stop_loader()
                await _start_loader(new_message=True)
            if waiting_for_segment:
                await _stop_loader()
                waiting_for_segment = False
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
        logger.info(
            "telegram.stream suppressed_by_merge chat_id=%s thread_id=%s customer_id=%s",
            chat_id,
            thread_id,
            customer_id,
        )
        suppressed = True
    if next_chunk_task is not None and not next_chunk_task.done():
        next_chunk_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            async with asyncio.timeout(1.0):
                await next_chunk_task
    await _stop_loader()
    if not suppressed and not final_reply:
        logger.error(
            "telegram.stream no_final_reply chat_id=%s thread_id=%s customer_id=%s",
            chat_id,
            thread_id,
            customer_id,
        )
        stream_message_id = stream_state.get("message_id")
        fallback_text = (
            "I couldn't produce a visible user-facing reply for that step "
            "(the model/tool loop ended without displayable output)."
        )
        stream_message_id = (
            await client.upsert_stream_message(
                chat_id=chat_id,
                text=fallback_text,
                message_id=stream_message_id,
                parse_mode="HTML",
            )
            or stream_message_id
        )
        stream_state["message_id"] = stream_message_id
        final_reply = fallback_text
    logger.info(
        "telegram.stream complete chat_id=%s thread_id=%s customer_id=%s suppressed=%s final_chars=%s",
        chat_id,
        thread_id,
        customer_id,
        suppressed,
        len(str(final_reply or "")),
    )
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
