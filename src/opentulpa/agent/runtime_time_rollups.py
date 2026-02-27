"""Runtime helpers for live time context and thread rollup persistence."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from typing import Any


async def build_live_time_context(
    *,
    customer_id: str,
    load_user_utc_offset: Callable[[str], Awaitable[str | None]],
    minutes_to_utc_offset: Callable[[int], str],
    utc_offset_to_minutes: Callable[[str], int | None],
) -> dict[str, str]:
    now_server = datetime.now().astimezone()
    now_utc = datetime.now(timezone.utc)
    server_offset = now_server.utcoffset() or timedelta()
    server_offset_minutes = int(server_offset.total_seconds() // 60)
    server_offset_text = minutes_to_utc_offset(server_offset_minutes)

    user_offset_text = await load_user_utc_offset(customer_id)
    source = "profile"
    user_offset_minutes = (
        utc_offset_to_minutes(user_offset_text) if user_offset_text else None
    )
    if user_offset_minutes is None:
        user_offset_minutes = server_offset_minutes
        user_offset_text = server_offset_text
        source = "fallback_server_timezone"

    user_local = now_utc + timedelta(minutes=user_offset_minutes)
    return {
        "server_time_local_iso": now_server.isoformat(),
        "server_time_utc_iso": now_utc.isoformat(),
        "server_utc_offset": server_offset_text,
        "user_time_local_iso": user_local.isoformat(),
        "user_utc_offset": user_offset_text,
        "user_time_source": source,
    }


def cap_rollup_text(*, text: str | None, context_rollup_tokens: int) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    max_chars = max(800, int(context_rollup_tokens) * 4)
    if len(raw) <= max_chars:
        return raw
    reserve = max(200, max_chars // 2 - 8)
    return f"{raw[:reserve]}\n...\n{raw[-reserve:]}"


def load_thread_rollup(
    *,
    thread_id: str,
    thread_rollup_service: Any | None,
    context_rollup_tokens: int,
) -> str | None:
    tid = str(thread_id or "").strip()
    if not tid or thread_rollup_service is None:
        return None
    try:
        text = thread_rollup_service.get_rollup(tid)
        return cap_rollup_text(
            text=text,
            context_rollup_tokens=context_rollup_tokens,
        )
    except Exception:
        return None


def save_thread_rollup(
    *,
    thread_id: str,
    rollup: str,
    thread_rollup_service: Any | None,
    context_rollup_tokens: int,
) -> None:
    tid = str(thread_id or "").strip()
    text = cap_rollup_text(
        text=rollup,
        context_rollup_tokens=context_rollup_tokens,
    )
    if not tid or not text or thread_rollup_service is None:
        return
    with suppress(Exception):
        thread_rollup_service.set_rollup(tid, text)
