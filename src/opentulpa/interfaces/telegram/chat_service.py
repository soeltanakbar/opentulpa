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
from opentulpa.interfaces.telegram.chat_commands import (
    format_agent_error_for_user as _format_agent_error_for_user,
)
from opentulpa.interfaces.telegram.chat_commands import (
    handle_control_command as _handle_control_command,
)
from opentulpa.interfaces.telegram.chat_commands import (
    inject_voice_message_context as _inject_voice_message_context,
)
from opentulpa.interfaces.telegram.client import parse_telegram_update
from opentulpa.interfaces.telegram.constants import STATE_PATH
from opentulpa.interfaces.telegram.env_management import missing_key_prompt
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
from opentulpa.interfaces.telegram.session_state import (
    ensure_admin_and_read_pending as _ensure_admin_and_read_pending,
)
from opentulpa.interfaces.telegram.session_state import (
    upsert_session_for_chat as _upsert_session_for_chat,
)
from opentulpa.interfaces.telegram.state_store import TelegramStateStore

STATE_STORE = TelegramStateStore(STATE_PATH)
logger = logging.getLogger(__name__)

def find_session_slots_for_customer_id(customer_id: str) -> list[dict[str, Any]]:
    return STATE_STORE.find_session_slots(customer_id)


def get_session_slot_for_chat_id(chat_id: int) -> dict[str, Any] | None:
    return STATE_STORE.get_session_slot(chat_id)


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

    admin_user_id, pending_key = STATE_STORE.update(
        lambda state: _ensure_admin_and_read_pending(
            state,
            chat_id=ctx.chat_id,
            user_id=ctx.user_id,
        )
    )
    is_admin = int(admin_user_id) == int(ctx.user_id)

    control_response = _handle_control_command(
        text=ctx.text,
        chat_id=ctx.chat_id,
        user_id=ctx.user_id,
        is_admin=is_admin,
        pending_key=pending_key,
        state_store=STATE_STORE,
        agent_runtime=agent_runtime,
    )
    if control_response is not None:
        return control_response

    if not os.environ.get("OPENROUTER_API_KEY"):
        return missing_key_prompt()
    if agent_runtime is None:
        return "Agent runtime is unavailable. Restart OpenTulpa and try again."

    thread_id, customer_id = STATE_STORE.update(
        lambda state: _upsert_session_for_chat(
            state,
            chat_id=ctx.chat_id,
            user_id=ctx.user_id,
        )
    )

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

    voice_transcripts = [
        str(item.get("voice_transcript", "")).strip()
        for item in ingested_files
        if str(item.get("kind", "")).strip() == "voice"
    ]
    non_voice_files = [
        item for item in ingested_files if str(item.get("kind", "")).strip() != "voice"
    ]

    context_blob = build_uploaded_files_context(non_voice_files)
    effective_text = _inject_voice_message_context(ctx.text, voice_transcripts)
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
        has_voice = any(str(getattr(item, "kind", "")).strip() == "voice" for item in attachments)
        if has_voice:
            return (
                "I received your voice message but couldn't transcribe it. "
                "Please resend a shorter/clearer voice note or send text."
            )
        return None

    if bot_token:
        try:
            final, suppressed = await stream_langgraph_reply_to_telegram(
                agent_runtime=agent_runtime,
                thread_id=thread_id,
                customer_id=customer_id,
                text=effective_text,
                bot_token=bot_token,
                chat_id=ctx.chat_id,
            )
            if suppressed:
                return None
        except Exception as exc:
            logger.exception(
                "Telegram streaming reply failed (chat_id=%s, thread_id=%s): %s",
                ctx.chat_id,
                thread_id,
                exc,
            )
            return _format_agent_error_for_user(exc)
        if final:
            STATE_STORE.touch_assistant_message(ctx.chat_id)
            return None
        debug_log(
            hypothesis_id="H4",
            location="interfaces/telegram/chat_service.py:handle_telegram_text",
            message="fallback_no_final_reply",
            data={"chat_id": ctx.chat_id, "thread_id": thread_id},
        )
        return "I received your message but no final reply was available yet. Ask again or use /status."

    try:
        response = await agent_runtime.ainvoke_text(
            thread_id=thread_id,
            customer_id=customer_id,
            text=effective_text,
        )
        return response
    except Exception as exc:
        logger.exception(
            "Telegram non-streaming reply failed (chat_id=%s, thread_id=%s): %s",
            ctx.chat_id,
            thread_id,
            exc,
        )
        return _format_agent_error_for_user(exc)


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

    def get_session_slot(self, chat_id: int) -> dict[str, Any] | None:
        return get_session_slot_for_chat_id(chat_id)

    def touch_assistant_message(self, chat_id: int) -> None:
        STATE_STORE.touch_assistant_message(chat_id)

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
