"""Telegram chat bridge orchestration service."""

from __future__ import annotations

import logging
import os
from typing import Any

from opentulpa.context.file_vault import FileVaultService
from opentulpa.interfaces.telegram.attachments import (
    build_uploaded_files_context,
    extract_attachments,
    ingest_attachments,
)
from opentulpa.interfaces.telegram.client import parse_telegram_update
from opentulpa.interfaces.telegram.constants import STATE_PATH
from opentulpa.interfaces.telegram.env_management import (
    extract_inline_key_value,
    extract_set_command,
    mask_secret,
    missing_key_prompt,
    status_text,
    upsert_env_key,
)
from opentulpa.interfaces.telegram.models import TelegramContext
from opentulpa.interfaces.telegram.relay import (
    debug_log,
    stream_langgraph_reply_to_telegram,
)
from opentulpa.interfaces.telegram.relay import (
    relay_event_via_main_agent as _relay_event_via_main_agent,
)
from opentulpa.interfaces.telegram.relay import (
    relay_task_event_via_main_agent as _relay_task_event_via_main_agent,
)
from opentulpa.interfaces.telegram.security import is_user_allowed
from opentulpa.interfaces.telegram.state_store import TelegramStateStore

STATE_STORE = TelegramStateStore(STATE_PATH)
logger = logging.getLogger(__name__)


def find_session_slots_for_customer_id(customer_id: str) -> list[dict[str, Any]]:
    return STATE_STORE.find_session_slots(customer_id)


def _find_session_slots_for_customer_id(customer_id: str) -> list[dict[str, Any]]:
    """Backward-compatible alias."""
    return find_session_slots_for_customer_id(customer_id)


async def relay_task_event_via_main_agent(
    *,
    customer_id: str,
    task_id: str,
    event_type: str,
    payload: dict[str, Any],
    agent_runtime: Any | None = None,
) -> list[dict[str, Any]]:
    return await _relay_task_event_via_main_agent(
        customer_id=customer_id,
        task_id=task_id,
        event_type=event_type,
        payload=payload,
        state_store=STATE_STORE,
        find_session_slots=find_session_slots_for_customer_id,
        agent_runtime=agent_runtime,
    )


async def relay_event_via_main_agent(
    *,
    customer_id: str,
    event_label: str,
    payload: dict[str, Any],
    agent_runtime: Any | None = None,
) -> list[dict[str, Any]]:
    return await _relay_event_via_main_agent(
        customer_id=customer_id,
        event_label=event_label,
        payload=payload,
        state_store=STATE_STORE,
        find_session_slots=find_session_slots_for_customer_id,
        agent_runtime=agent_runtime,
    )


async def handle_telegram_text(
    *,
    body: dict[str, Any],
    bot_token: str | None = None,
    allowed_user_ids_csv: str | None = None,
    allowed_usernames_csv: str | None = None,
    agent_runtime: Any | None = None,
    file_vault: FileVaultService | None = None,
    memory: Any | None = None,
) -> str | None:
    parsed = parse_telegram_update(body)
    if not parsed:
        return None
    chat_id, user_id, text = parsed
    if not chat_id or not user_id:
        return None

    message = body.get("message") or body.get("edited_message") or {}
    caption = str(message.get("caption", "")).strip() or None
    attachments = extract_attachments(message)
    username = (message.get("from", {}) or {}).get("username")
    username = username.strip() or None if isinstance(username, str) else None
    ctx = TelegramContext(
        chat_id=chat_id,
        user_id=user_id,
        username=username,
        text=(text or "").strip(),
    )

    if not is_user_allowed(
        user_id=ctx.user_id,
        username=ctx.username,
        allowed_user_ids_csv=allowed_user_ids_csv,
        allowed_usernames_csv=allowed_usernames_csv,
    ):
        return "This bot is restricted and your Telegram account is not allowed."

    state = STATE_STORE.load()
    admin_user_id = state.get("admin_user_id")
    if admin_user_id is None:
        state["admin_user_id"] = ctx.user_id
        admin_user_id = ctx.user_id
        STATE_STORE.save(state)
    is_admin = int(admin_user_id) == int(ctx.user_id)
    pending = state.get("pending_key_by_chat", {})
    pending_key = pending.get(str(ctx.chat_id))

    text_lower = ctx.text.lower()
    if text_lower in {"/start", "/help"}:
        return (
            "OpenTulpa Telegram is connected.\n"
            "I can chat and help with key setup.\n\n"
            "Commands:\n"
            "/status\n"
            "/setup\n"
            "/set KEY VALUE\n"
            "/setenv KEY VALUE\n"
            "/cancel"
        )
    if text_lower == "/status":
        agent_up = bool(agent_runtime and getattr(agent_runtime, "healthy", lambda: False)())
        return status_text(agent_up)
    if text_lower == "/cancel":
        pending.pop(str(ctx.chat_id), None)
        state["pending_key_by_chat"] = pending
        STATE_STORE.save(state)
        return "Cancelled pending setup."

    if text_lower == "/setup":
        if not is_admin:
            return "Only the bot admin can set keys."
        if not os.environ.get("OPENROUTER_API_KEY"):
            pending[str(ctx.chat_id)] = "OPENROUTER_API_KEY"
            state["pending_key_by_chat"] = pending
            STATE_STORE.save(state)
            return "Please send your OPENROUTER_API_KEY value now."
        return "Core key is already set. Use /set KEY VALUE for additional keys."

    if pending_key:
        if not is_admin:
            return "Only the bot admin can set keys."
        value = ctx.text.strip()
        if not value:
            return f"Please send the value for {pending_key}."
        try:
            upsert_env_key(pending_key, value)
        except Exception as exc:
            return f"Failed to save {pending_key}: {exc}"
        pending.pop(str(ctx.chat_id), None)
        state["pending_key_by_chat"] = pending
        STATE_STORE.save(state)
        return f"Saved {pending_key}={mask_secret(value)}.\nRestart OpenTulpa to apply."

    kv = extract_set_command(ctx.text) or extract_inline_key_value(ctx.text)
    if kv:
        key, value = kv
        if not is_admin:
            return "Only the bot admin can set keys."
        try:
            upsert_env_key(key, value)
        except Exception as exc:
            return f"Failed to save {key}: {exc}"
        return f"Saved {key}={mask_secret(value)}."

    if not os.environ.get("OPENROUTER_API_KEY"):
        return missing_key_prompt()
    if agent_runtime is None:
        return "Agent runtime is unavailable. Restart OpenTulpa and try again."

    sessions = state.get("sessions", {})
    slot = sessions.get(str(ctx.chat_id), {})
    thread_id = str(slot.get("thread_id", "")).strip() or f"chat-{ctx.chat_id}"
    customer_id = str(slot.get("customer_id", "")).strip() or f"telegram_{ctx.user_id}"
    sessions[str(ctx.chat_id)] = {
        "user_id": ctx.user_id,
        "customer_id": customer_id,
        "thread_id": thread_id,
        "wake_thread_id": slot.get("wake_thread_id"),
    }
    state["sessions"] = sessions
    STATE_STORE.save(state)

    ingested_files: list[dict[str, Any]] = []
    if attachments and bot_token and file_vault is not None:
        ingested_files = await ingest_attachments(
            attachments=attachments,
            bot_token=bot_token,
            file_vault=file_vault,
            memory=memory,
            agent_runtime=agent_runtime,
            customer_id=customer_id,
            chat_id=ctx.chat_id,
            caption=caption,
        )

    if attachments and not ctx.text and not ingested_files:
        if agent_runtime is None:
            return "I received your file, but agent runtime is unavailable right now."
        if file_vault is None:
            return "I received your file, but file storage is not configured."

    context_blob = build_uploaded_files_context(ingested_files)
    effective_text = ctx.text
    if context_blob:
        if effective_text:
            effective_text = f"{effective_text}\n\n{context_blob}"
        else:
            effective_text = (
                "User uploaded one or more files without extra text.\n"
                "Summarize what is available and ask a focused follow-up question.\n\n"
                f"{context_blob}"
            )

    if not effective_text:
        return None

    if bot_token:
        final = await stream_langgraph_reply_to_telegram(
            agent_runtime=agent_runtime,
            thread_id=thread_id,
            customer_id=customer_id,
            text=effective_text,
            bot_token=bot_token,
            chat_id=ctx.chat_id,
        )
        if final:
            return None
        debug_log(
            hypothesis_id="H4",
            location="interfaces/telegram/chat_service.py:handle_telegram_text",
            message="fallback_no_final_reply",
            data={"chat_id": ctx.chat_id, "thread_id": thread_id},
        )
        return "I received your message but no final reply was available yet. Ask again or use /status."

    response = await agent_runtime.ainvoke_text(
        thread_id=thread_id,
        customer_id=customer_id,
        text=effective_text,
    )
    return response


class TelegramChatService:
    """Telegram chat orchestration service with injected dependencies."""

    def __init__(
        self,
        *,
        bot_token: str,
        file_vault: FileVaultService | None = None,
        memory: Any | None = None,
    ) -> None:
        self.bot_token = str(bot_token or "").strip()
        self.file_vault = file_vault
        self.memory = memory

    def find_session_slots(self, customer_id: str) -> list[dict[str, Any]]:
        return find_session_slots_for_customer_id(customer_id)

    async def relay_task_event(
        self,
        *,
        customer_id: str,
        task_id: str,
        event_type: str,
        payload: dict[str, Any],
        agent_runtime: Any | None = None,
    ) -> list[dict[str, Any]]:
        return await relay_task_event_via_main_agent(
            customer_id=customer_id,
            task_id=task_id,
            event_type=event_type,
            payload=payload,
            agent_runtime=agent_runtime,
        )

    async def relay_event(
        self,
        *,
        customer_id: str,
        event_label: str,
        payload: dict[str, Any],
        agent_runtime: Any | None = None,
    ) -> list[dict[str, Any]]:
        return await relay_event_via_main_agent(
            customer_id=customer_id,
            event_label=event_label,
            payload=payload,
            agent_runtime=agent_runtime,
        )

    async def handle_update(
        self,
        *,
        body: dict[str, Any],
        allowed_user_ids_csv: str | None = None,
        allowed_usernames_csv: str | None = None,
        agent_runtime: Any | None = None,
    ) -> str | None:
        return await handle_telegram_text(
            body=body,
            bot_token=self.bot_token,
            allowed_user_ids_csv=allowed_user_ids_csv,
            allowed_usernames_csv=allowed_usernames_csv,
            agent_runtime=agent_runtime,
            file_vault=self.file_vault,
            memory=self.memory,
        )
