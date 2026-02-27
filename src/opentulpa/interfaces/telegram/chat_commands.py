"""Telegram control/setup command helpers."""

from __future__ import annotations

import os
from typing import Any

from opentulpa.interfaces.telegram.env_management import (
    extract_inline_key_value,
    extract_set_command,
    mask_secret,
    status_text,
    upsert_env_key,
)
from opentulpa.interfaces.telegram.session_state import (
    clear_pending_for_chat as _clear_pending_for_chat,
)
from opentulpa.interfaces.telegram.session_state import (
    reset_chat_session_context as _reset_chat_session_context,
)
from opentulpa.interfaces.telegram.session_state import (
    set_pending_key_for_chat as _set_pending_key_for_chat,
)


def format_agent_error_for_user(exc: Exception) -> str:
    """Convert backend/model failures into actionable Telegram-safe user messages."""
    text = str(exc)
    lowered = text.lower()
    if "401" in lowered and (
        "user not found" in lowered
        or "authentication" in lowered
        or "invalid api key" in lowered
        or "unauthorized" in lowered
    ):
        return (
            "Model authentication failed (OpenRouter key is invalid or revoked). "
            "Set a valid OPENROUTER_API_KEY and restart OpenTulpa."
        )
    if "429" in lowered or "rate limit" in lowered:
        return "The model provider is rate-limiting requests right now. Please try again shortly."
    return "I hit a backend error while generating a reply. Please try again."


def inject_voice_message_context(text: str, transcripts: list[str]) -> str:
    safe_lines = [str(item).strip() for item in transcripts if str(item).strip()]
    if not safe_lines:
        return str(text or "")
    voice_block = "\n".join(f"<user sent voice message>: {line}" for line in safe_lines)
    base = str(text or "").strip()
    if base:
        return f"{base}\n\n{voice_block}"
    return voice_block


def start_help_text() -> str:
    return (
        "OpenTulpa is connected.\n\n"
        "What I can do:\n"
        "- Web + links: web search, read URLs, summarize current info\n"
        "- Interactive browsing: browser automation for dynamic sites (when configured)\n"
        "- Files: analyze PDFs/DOCX/text/images/voice notes you send\n"
        "- Code + automations: write/debug scripts, run checks, schedule recurring tasks\n"
        "- Memory + preferences: remember your style/process directives\n\n"
        "To personalize quickly, answer these:\n"
        "1. What are you struggling with right now?\n"
        "2. Which repetitive task should I automate first?\n"
        "3. Which services should I connect first (Gmail, Sheets, custom APIs, etc.)?\n\n"
        "Commands:\n"
        "/status\n"
        "/setup\n"
        "/set KEY VALUE\n"
        "/setenv KEY VALUE\n"
        "/fresh\n"
        "/cancel"
    )


def handle_control_command(
    *,
    text: str,
    chat_id: int,
    user_id: int,
    is_admin: bool,
    pending_key: str | None,
    state_store: Any,
    agent_runtime: Any | None,
) -> str | None:
    text_lower = str(text or "").strip().lower()
    if text_lower in {"/start", "/help"}:
        return start_help_text()
    if text_lower == "/status":
        agent_up = bool(agent_runtime and getattr(agent_runtime, "healthy", lambda: False)())
        return status_text(agent_up)
    if text_lower == "/cancel":
        state_store.update(lambda state: _clear_pending_for_chat(state, chat_id=chat_id))
        return "Cancelled pending setup."
    if text_lower == "/fresh":
        thread_id, _ = state_store.update(
            lambda state: _reset_chat_session_context(
                state,
                chat_id=chat_id,
                user_id=user_id,
            )
        )
        return (
            "Started a fresh chat context. "
            f"New thread: {thread_id}. "
            "Your long-term memory is unchanged."
        )

    if text_lower == "/setup":
        if not is_admin:
            return "Only the bot admin can set keys."
        if not os.environ.get("OPENROUTER_API_KEY"):
            state_store.update(
                lambda state: _set_pending_key_for_chat(
                    state,
                    chat_id=chat_id,
                    key="OPENROUTER_API_KEY",
                )
            )
            return "Please send your OPENROUTER_API_KEY value now."
        return "Core key is already set. Use /set KEY VALUE for additional keys."

    if pending_key:
        if not is_admin:
            return "Only the bot admin can set keys."
        value = str(text or "").strip()
        if not value:
            return f"Please send the value for {pending_key}."
        try:
            upsert_env_key(pending_key, value)
        except Exception as exc:
            return f"Failed to save {pending_key}: {exc}"

        state_store.update(lambda state: _clear_pending_for_chat(state, chat_id=chat_id))
        return f"Saved {pending_key}={mask_secret(value)}.\nRestart OpenTulpa to apply."

    kv = extract_set_command(text) or extract_inline_key_value(text)
    if kv:
        key, value = kv
        if not is_admin:
            return "Only the bot admin can set keys."
        try:
            upsert_env_key(key, value)
        except Exception as exc:
            return f"Failed to save {key}: {exc}"
        return f"Saved {key}={mask_secret(value)}."
    return None
