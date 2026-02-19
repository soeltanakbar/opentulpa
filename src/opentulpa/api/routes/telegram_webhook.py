"""Telegram webhook route registration."""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import JSONResponse

from opentulpa.interfaces.telegram.client import (
    parse_telegram_callback_query,
    parse_telegram_update,
)

logger = logging.getLogger(__name__)


def _parse_approval_callback_data(data: str) -> tuple[str, str] | None:
    raw = str(data or "").strip()
    # Format: approval:<approval_id>:approve|deny
    match = re.fullmatch(r"approval:([a-z0-9_-]{6,40}):(approve|deny)", raw, re.IGNORECASE)
    if not match:
        return None
    return match.group(1), match.group(2).lower()


def _parse_text_token_decision(text: str) -> tuple[str, str] | None:
    raw = str(text or "").strip()
    # Formats: /approve apr_xxx | approve apr_xxx | /deny apr_xxx | deny apr_xxx
    match = re.fullmatch(r"/?(approve|deny)\s+([a-z0-9_-]{6,40})", raw, re.IGNORECASE)
    if not match:
        return None
    return match.group(2), match.group(1).lower()


def register_telegram_webhook_routes(
    app: FastAPI,
    *,
    settings: Any,
    get_telegram_client: Callable[[], Any],
    get_telegram_chat: Callable[[], Any],
    get_agent_runtime: Callable[[], Any],
    decide_approval_and_maybe_wake: Callable[..., Awaitable[dict[str, Any]]],
) -> None:
    """Register Telegram webhook with callback + text-token approval support."""

    @app.post("/webhook/telegram")
    async def telegram_webhook(request: Request, background_tasks: BackgroundTasks) -> Response:
        if not settings.telegram_bot_token:
            return JSONResponse(status_code=501, content={"detail": "Telegram not configured"})
        if settings.telegram_webhook_secret:
            incoming_secret = request.headers.get("x-telegram-bot-api-secret-token", "")
            if incoming_secret != settings.telegram_webhook_secret:
                return JSONResponse(status_code=403, content={"detail": "invalid telegram secret"})
        body = await request.json()

        # Immediate 200 OK, logic runs in background.
        background_tasks.add_task(_telegram_background_handler, body=body)
        return Response(status_code=200)

    async def _telegram_background_handler(body: dict[str, Any]) -> None:
        callback_id, callback_user_id, callback_chat_id, callback_data, callback_message_id = (
            parse_telegram_callback_query(body)
        )
        parsed_callback = _parse_approval_callback_data(callback_data or "")
        if parsed_callback and callback_id and callback_user_id and callback_chat_id:
            approval_id, decision = parsed_callback
            result = await decide_approval_and_maybe_wake(
                approval_id=approval_id,
                decision=decision,
                actor_interface="telegram",
                actor_id=str(callback_user_id),
            )
            ack_text = (
                f"Approval {decision}d."
                if bool(result.get("ok"))
                else f"Approval {decision} failed: {result.get('reason', 'not_allowed')}"
            )
            with suppress(Exception):
                await get_telegram_client().answer_callback_query(
                    callback_query_id=callback_id,
                    text=ack_text,
                    show_alert=False,
                )
            if isinstance(callback_message_id, int):
                with suppress(Exception):
                    await get_telegram_client().edit_message_text(
                        chat_id=callback_chat_id,
                        message_id=callback_message_id,
                        text=ack_text,
                        parse_mode="HTML",
                        reply_markup={"inline_keyboard": []},
                    )
            return

        message = body.get("message") or body.get("edited_message") or {}
        chat_id = message.get("chat", {}).get("id")
        _, user_id, text = parse_telegram_update(body)
        token_decision = _parse_text_token_decision(text or "")
        if token_decision and chat_id is not None and user_id is not None:
            approval_id, decision = token_decision
            result = await decide_approval_and_maybe_wake(
                approval_id=approval_id,
                decision=decision,
                actor_interface="telegram",
                actor_id=str(user_id),
            )
            reply_text = (
                f"Approval {decision}d."
                if bool(result.get("ok"))
                else f"Approval {decision} failed: {result.get('reason', 'not_allowed')}"
            )
            with suppress(Exception):
                await get_telegram_client().send_message(
                    chat_id=chat_id,
                    text=reply_text,
                    parse_mode="HTML",
                )
            return

        try:
            reply = await get_telegram_chat().handle_update(
                body=body,
                allowed_user_ids_csv=settings.telegram_allowed_user_ids,
                allowed_usernames_csv=settings.telegram_allowed_usernames,
                agent_runtime=get_agent_runtime(),
            )
        except Exception as exc:
            logger.exception("Unhandled Telegram background handler failure: %s", exc)
            if chat_id is not None:
                with suppress(Exception):
                    await get_telegram_client().send_message(
                        chat_id=chat_id,
                        text="I hit an internal error while processing your message. Please try again.",
                        parse_mode="HTML",
                    )
            return

        if reply and chat_id is not None:
            with suppress(Exception):
                await get_telegram_client().send_message(
                    chat_id=chat_id,
                    text=reply,
                    parse_mode="HTML",
                )
