"""Telegram reply streaming and wake-event relays."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from opentulpa.core.ids import new_short_id
from opentulpa.interfaces.telegram.client import TelegramClient
from opentulpa.interfaces.telegram.constants import DEBUG_LOG_PATH, LOW_SIGNAL_REPLIES


def normalize_reply_text(text: str) -> str:
    import re

    t = text.strip().lower()
    t = re.sub(r"[.!?]+$", "", t)
    return " ".join(t.split())


def is_low_signal_reply(text: str) -> bool:
    return normalize_reply_text(text) in LOW_SIGNAL_REPLIES


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


async def stream_langgraph_reply_to_telegram(
    *,
    agent_runtime: Any,
    thread_id: str,
    customer_id: str,
    text: str,
    bot_token: str,
    chat_id: int,
) -> str | None:
    stream_message_id: int | None = None
    last_streamed = ""
    final_reply = None
    client = TelegramClient(bot_token)
    async for partial in agent_runtime.astream_text(
        thread_id=thread_id,
        customer_id=customer_id,
        text=text,
    ):
        if not isinstance(partial, str):
            continue
        current = partial.strip()
        if not current or is_low_signal_reply(current) or current == last_streamed:
            continue
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
    return final_reply


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
        state = state_store.load()
        sessions = state.get("sessions", {})
        raw_slot = sessions.get(str(chat_id), {}) if isinstance(sessions, dict) else {}
        wake_thread_id = str(raw_slot.get("wake_thread_id", "")).strip()
        if not wake_thread_id:
            wake_thread_id = new_short_id("wake")
            if isinstance(raw_slot, dict):
                raw_slot["wake_thread_id"] = wake_thread_id
                sessions[str(chat_id)] = raw_slot
                state["sessions"] = sessions
                state_store.save(state)
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
