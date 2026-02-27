"""Telegram reply streaming and wake-event relays."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any

from opentulpa.agent.runtime import (
    STREAM_APPROVAL_HANDOFF_SIGNAL,
    STREAM_WAIT_SIGNAL,
)
from opentulpa.interfaces.telegram.client import TelegramClient
from opentulpa.interfaces.telegram.constants import DEBUG_LOG_PATH, LOW_SIGNAL_REPLIES
from opentulpa.interfaces.telegram.relay_events import (
    relay_event_via_main_agent as _relay_event_via_main_agent,
)
from opentulpa.interfaces.telegram.relay_events import (
    relay_task_event_via_main_agent as _relay_task_event_via_main_agent,
)
from opentulpa.interfaces.telegram.relay_streaming import (
    _emit_typing_until_done as _emit_typing_until_done_impl,
)
from opentulpa.interfaces.telegram.relay_streaming import (
    stream_langgraph_reply_to_telegram as _stream_langgraph_reply_to_telegram,
)

NO_NOTIFY_TOKEN = "__NO_NOTIFY__"


async def _emit_typing_until_done(*, client: Any, chat_id: int, stop_event: Any) -> None:
    await _emit_typing_until_done_impl(client=client, chat_id=chat_id, stop_event=stop_event)


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


async def stream_langgraph_reply_to_telegram(
    *,
    agent_runtime: Any,
    thread_id: str,
    customer_id: str,
    text: str,
    bot_token: str,
    chat_id: int,
) -> tuple[str | None, bool]:
    return await _stream_langgraph_reply_to_telegram(
        agent_runtime=agent_runtime,
        thread_id=thread_id,
        customer_id=customer_id,
        text=text,
        bot_token=bot_token,
        chat_id=chat_id,
        is_low_signal_reply=is_low_signal_reply,
        stream_wait_signal=STREAM_WAIT_SIGNAL,
        stream_approval_handoff_signal=STREAM_APPROVAL_HANDOFF_SIGNAL,
        telegram_client_factory=TelegramClient,
        asyncio_module=asyncio,
    )


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
    return await _relay_task_event_via_main_agent(
        customer_id=customer_id,
        task_id=task_id,
        event_type=event_type,
        payload=payload,
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
    return await _relay_event_via_main_agent(
        customer_id=customer_id,
        event_label=event_label,
        payload=payload,
        state_store=state_store,
        find_session_slots=find_session_slots,
        agent_runtime=agent_runtime,
        no_notify_token=NO_NOTIFY_TOKEN,
    )
