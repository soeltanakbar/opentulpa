"""Telegram session-state helpers (pure state mutations)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from opentulpa.core.ids import new_short_id


def clean_thread_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"none", "null"}:
        return ""
    return text


def ensure_admin_and_read_pending(
    state: dict[str, Any],
    *,
    chat_id: int,
    user_id: int,
) -> tuple[int, str | None]:
    admin_user_id = state.get("admin_user_id")
    if admin_user_id is None:
        admin_user_id = user_id
        state["admin_user_id"] = admin_user_id
    pending_map = state.get("pending_key_by_chat")
    if not isinstance(pending_map, dict):
        pending_map = {}
        state["pending_key_by_chat"] = pending_map
    pending_key = pending_map.get(str(chat_id))
    if pending_key is None:
        return int(admin_user_id), None
    pending_key_text = str(pending_key).strip()
    return int(admin_user_id), (pending_key_text or None)


def clear_pending_for_chat(state: dict[str, Any], *, chat_id: int) -> None:
    pending_map = state.get("pending_key_by_chat")
    if not isinstance(pending_map, dict):
        pending_map = {}
    pending_map.pop(str(chat_id), None)
    state["pending_key_by_chat"] = pending_map


def set_pending_key_for_chat(state: dict[str, Any], *, chat_id: int, key: str) -> None:
    pending_map = state.get("pending_key_by_chat")
    if not isinstance(pending_map, dict):
        pending_map = {}
    pending_map[str(chat_id)] = str(key).strip()
    state["pending_key_by_chat"] = pending_map


def reset_chat_session_context(
    state: dict[str, Any],
    *,
    chat_id: int,
    user_id: int,
) -> tuple[str, str]:
    sessions = state.get("sessions")
    if not isinstance(sessions, dict):
        sessions = {}
    chat_key = str(chat_id)
    slot = sessions.get(chat_key)
    if not isinstance(slot, dict):
        slot = {}
    customer_id = str(slot.get("customer_id", "")).strip() or f"telegram_{user_id}"
    now_utc_iso = datetime.now(timezone.utc).isoformat()
    thread_id = new_short_id("chat")
    wake_thread_id = new_short_id("wake")
    sessions[chat_key] = {
        "user_id": int(user_id),
        "customer_id": customer_id,
        "thread_id": thread_id,
        "wake_thread_id": wake_thread_id,
        "last_user_message_at": now_utc_iso,
        "last_assistant_message_at": None,
    }
    state["sessions"] = sessions
    clear_pending_for_chat(state, chat_id=chat_id)
    return thread_id, customer_id


def upsert_session_for_chat(
    state: dict[str, Any],
    *,
    chat_id: int,
    user_id: int,
) -> tuple[str, str]:
    sessions = state.get("sessions")
    if not isinstance(sessions, dict):
        sessions = {}
    slot = sessions.get(str(chat_id))
    if not isinstance(slot, dict):
        slot = {}
    thread_id = clean_thread_id(slot.get("thread_id")) or f"chat-{chat_id}"
    wake_thread_id = clean_thread_id(slot.get("wake_thread_id")) or None
    customer_id = str(slot.get("customer_id", "")).strip() or f"telegram_{user_id}"
    now_utc_iso = datetime.now(timezone.utc).isoformat()
    sessions[str(chat_id)] = {
        "user_id": int(user_id),
        "customer_id": customer_id,
        "thread_id": thread_id,
        "wake_thread_id": wake_thread_id,
        "last_user_message_at": now_utc_iso,
        "last_assistant_message_at": slot.get("last_assistant_message_at"),
    }
    state["sessions"] = sessions
    return thread_id, customer_id
