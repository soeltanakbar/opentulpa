"""Wake/task event relay helpers for Telegram."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any

from opentulpa.agent.result_models import WakeEventDecision
from opentulpa.core.ids import new_short_id

NO_NOTIFY_TOKEN = "__NO_NOTIFY__"


def _clean_thread_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"none", "null"}:
        return ""
    return text


def _event_wake_scope(event_label: str, payload: dict[str, Any]) -> str:
    safe_label = str(event_label or "").strip().lower()
    nested_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}

    if safe_label.startswith("routine/"):
        routine_id = str(payload.get("routine_id") or nested_payload.get("routine_id") or "").strip()
        if routine_id:
            return f"routine:{routine_id}"
        return ""

    if safe_label.startswith("approval/"):
        approval_id = str(payload.get("approval_id") or nested_payload.get("approval_id") or "").strip()
        if approval_id:
            return f"approval:{approval_id}"
        return "approval"

    if safe_label.startswith("task/"):
        task_id = str(payload.get("task_id") or nested_payload.get("task_id") or "").strip()
        if task_id:
            return f"task:{task_id}"
        return "task"

    if safe_label:
        return safe_label.replace("/", ":")
    return ""


def _resolve_slot_wake_thread_id(*, raw_slot: dict[str, Any], scope: str) -> str:
    if not scope:
        return _clean_thread_id(raw_slot.get("wake_thread_id"))

    slot_threads = raw_slot.get("wake_thread_ids")
    if not isinstance(slot_threads, dict):
        slot_threads = {}
    scoped_id = _clean_thread_id(slot_threads.get(scope))
    return scoped_id


def _store_slot_wake_thread_id(
    *,
    raw_slot: dict[str, Any],
    scope: str,
    wake_thread_id: str,
) -> None:
    if not scope:
        raw_slot["wake_thread_id"] = wake_thread_id
        return

    slot_threads = raw_slot.get("wake_thread_ids")
    if not isinstance(slot_threads, dict):
        slot_threads = {}
    slot_threads[scope] = wake_thread_id
    raw_slot["wake_thread_ids"] = slot_threads

    if scope.startswith("routine:"):
        raw_slot["wake_thread_id"] = wake_thread_id


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
    no_notify_token: str = NO_NOTIFY_TOKEN,
) -> list[dict[str, Any]]:
    slots = find_session_slots(customer_id)
    if not slots:
        return []
    if agent_runtime is None:
        raise RuntimeError("Agent runtime unavailable for wake relay")
    routine_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
    routine_message = str(routine_payload.get("message", "")).strip()
    routine_name = str(payload.get("routine_name", "")).strip()
    proactive_heartbeat = bool(routine_payload.get("proactive_heartbeat", False))
    wake_scope = _event_wake_scope(event_label, payload)
    now_utc = datetime.now(timezone.utc)
    replies: list[dict[str, Any]] = []
    for slot in slots:
        chat_id = int(slot["chat_id"])
        chat_key = str(chat_id)
        last_user_at = str(slot.get("last_user_message_at", "")).strip()
        last_assistant_at = str(slot.get("last_assistant_message_at", "")).strip()
        user_idle_hours = "unknown"
        assistant_idle_hours = "unknown"
        if last_user_at:
            with suppress(Exception):
                parsed = datetime.fromisoformat(last_user_at.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                user_idle_hours = f"{max(0.0, (now_utc - parsed).total_seconds() / 3600.0):.2f}"
        if last_assistant_at:
            with suppress(Exception):
                parsed = datetime.fromisoformat(last_assistant_at.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                assistant_idle_hours = f"{max(0.0, (now_utc - parsed).total_seconds() / 3600.0):.2f}"

        if (
            str(event_label).startswith("routine/")
            and proactive_heartbeat
            and hasattr(agent_runtime, "classify_wake_event")
        ):
            precheck_payload = {
                "event_label": event_label,
                "routine_name": routine_name,
                "routine_payload": routine_payload,
                "last_user_message_at_utc": last_user_at or "unknown",
                "last_assistant_message_at_utc": last_assistant_at or "unknown",
                "user_idle_hours": user_idle_hours,
                "assistant_idle_hours": assistant_idle_hours,
            }
            decision = WakeEventDecision(notify_user=True)
            with suppress(Exception):
                decision = WakeEventDecision.from_any(
                    await agent_runtime.classify_wake_event(
                        customer_id=customer_id,
                        event_label="routine/heartbeat_precheck",
                        payload=precheck_payload,
                    )
                )
            if not decision.notify_user:
                continue

        allow_no_notify_token = bool(str(event_label).startswith("routine/") and proactive_heartbeat)

        if str(event_label).startswith("routine/"):
            if allow_no_notify_token:
                instruction = (
                    "System update: a scheduled routine woke you.\n"
                    "Decide if the user should be messaged right now.\n"
                    f"- event: {event_label}\n"
                    f"- routine_name: {routine_name or 'unnamed'}\n"
                    f"- routine_instruction: {routine_message[:3000] or '(none)'}\n"
                    f"- last_user_message_at_utc: {last_user_at or 'unknown'}\n"
                    f"- user_idle_hours: {user_idle_hours}\n"
                    f"- last_assistant_message_at_utc: {last_assistant_at or 'unknown'}\n"
                    f"- assistant_idle_hours: {assistant_idle_hours}\n"
                    f"- now_utc: {now_utc.isoformat()}\n"
                    f"- payload: {json.dumps(payload, ensure_ascii=False)[:4000]}\n\n"
                    f"If you decide to skip messaging this run, reply exactly: {no_notify_token}\n"
                    "If you decide to message, send one concise, natural message (no rigid status template)."
                )
            else:
                instruction = (
                    "System update: a scheduled routine woke you.\n"
                    "Execute the routine instruction first, then send a concise status message.\n"
                    "Use available tools as needed, and do not ask the user for approval during this run.\n"
                    f"- event: {event_label}\n"
                    f"- routine_name: {routine_name or 'unnamed'}\n"
                    f"- routine_instruction: {routine_message[:3000] or '(none)'}\n"
                    f"- last_user_message_at_utc: {last_user_at or 'unknown'}\n"
                    f"- user_idle_hours: {user_idle_hours}\n"
                    f"- last_assistant_message_at_utc: {last_assistant_at or 'unknown'}\n"
                    f"- assistant_idle_hours: {assistant_idle_hours}\n"
                    f"- now_utc: {now_utc.isoformat()}\n"
                    f"- payload: {json.dumps(payload, ensure_ascii=False)[:4000]}\n\n"
                    "After execution, send one concise, natural update."
                )
        else:
            instruction = (
                "System update: a background event occurred.\n"
                "Respond with concise plain-language status, what happened, and next action.\n"
                f"- event: {event_label}\n"
                f"- payload: {json.dumps(payload, ensure_ascii=False)[:4000]}"
            )

        def _ensure_wake_thread_id(state: dict[str, Any], _chat_key: str = chat_key) -> str:
            sessions = state.get("sessions")
            if not isinstance(sessions, dict):
                sessions = {}
            raw_slot = sessions.get(_chat_key)
            if not isinstance(raw_slot, dict):
                raw_slot = {}
            wake_thread_id = _resolve_slot_wake_thread_id(raw_slot=raw_slot, scope=wake_scope)
            if not wake_thread_id or not wake_thread_id.lower().startswith("wake_"):
                wake_thread_id = new_short_id("wake")
                _store_slot_wake_thread_id(
                    raw_slot=raw_slot,
                    scope=wake_scope,
                    wake_thread_id=wake_thread_id,
                )
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
                recursion_limit_override=36 if proactive_heartbeat else None,
            )
            safe = str(text or "").strip()
            if not safe:
                continue
            if safe == no_notify_token and allow_no_notify_token:
                replies.append({"chat_id": chat_id, "text": no_notify_token})
                continue
            if safe == no_notify_token:
                continue
            replies.append({"chat_id": chat_id, "text": safe})
        except Exception:
            continue
    return replies
