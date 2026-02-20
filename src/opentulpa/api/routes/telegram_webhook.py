"""Telegram webhook route registration."""

from __future__ import annotations

import asyncio
import json
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


async def _execute_approved_action_and_summarize(
    *,
    get_agent_runtime: Callable[[], Any],
    get_context_events: Callable[[], Any],
    approval_id: str,
    decision_payload: dict[str, Any],
    chat_id: int,
) -> str:
    customer_id = str(decision_payload.get("customer_id", "")).strip()
    thread_id = str(decision_payload.get("thread_id", "")).strip() or f"chat-{chat_id}"
    if not customer_id:
        return "I approved this action, but couldn't resolve the customer context to execute it."
    runtime = get_agent_runtime()
    if runtime is None:
        return "I approved this action, but runtime is unavailable right now."
    if not hasattr(runtime, "execute_tool"):
        return "I approved this action, but runtime cannot execute approved actions."

    try:
        execution_result = await runtime.execute_tool(
            action_name="guardrail_execute_approved_action",
            action_args={"approval_id": approval_id, "customer_id": customer_id},
        )
    except Exception as exc:
        with suppress(Exception):
            get_context_events().add_event(
                customer_id=customer_id,
                source="approval",
                event_type="execute_failed",
                payload={
                    "approval_id": approval_id,
                    "thread_id": thread_id,
                    "error": str(exc),
                },
            )
        return f"I couldn't execute the approved action: {exc}"

    with suppress(Exception):
        get_context_events().add_event(
            customer_id=customer_id,
            source="approval",
            event_type="executed",
            payload={
                "approval_id": approval_id,
                "thread_id": thread_id,
                "execution_result": execution_result if isinstance(execution_result, dict) else {"raw": str(execution_result)},
            },
        )

    if isinstance(execution_result, dict) and bool(execution_result.get("already_executed")):
        return "This approved action was already executed successfully earlier."

    payload_preview = json.dumps(execution_result, ensure_ascii=False)[:6000]
    try:
        summary = await runtime.ainvoke_text(
            thread_id=thread_id,
            customer_id=customer_id,
            text=(
                "A previously approved action has just been executed.\n"
                f"approval_id={approval_id}\n"
                f"execution_result={payload_preview}\n\n"
                "Write a concise user-facing outcome message in plain text:\n"
                "1) what was done,\n"
                "2) whether it succeeded,\n"
                "3) next step only if needed.\n"
                "Do not expose internal JSON or system internals."
            ),
            include_pending_context=False,
        )
    except Exception:
        summary = ""

    final = str(summary or "").strip()
    if final:
        return final
    if isinstance(execution_result, dict) and str(execution_result.get("error", "")).strip():
        return f"I couldn't complete the approved action: {execution_result.get('error')}"
    return "Task completed."


async def _animate_loader_until_done(
    *,
    client: Any,
    chat_id: int,
    state: dict[str, Any],
    stop_event: asyncio.Event,
) -> None:
    sequence = ("...", "..", ".", "..", "...")
    idx = 0
    while not stop_event.is_set():
        marker = sequence[idx % len(sequence)]
        idx += 1
        state["message_id"] = (
            await client.upsert_stream_message(
                chat_id=chat_id,
                text=marker,
                message_id=state.get("message_id"),
                parse_mode=None,
                allow_fallback_send=False,
            )
            or state.get("message_id")
        )
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=0.22)


async def _run_post_approval_execution_flow(
    *,
    get_telegram_client: Callable[[], Any],
    get_agent_runtime: Callable[[], Any],
    get_context_events: Callable[[], Any],
    approval_id: str,
    decision_payload: dict[str, Any],
    chat_id: int,
    approval_message_id: int | None = None,
) -> None:
    client = get_telegram_client()
    with suppress(Exception):
        if isinstance(approval_message_id, int):
            await client.edit_message_text(
                chat_id=chat_id,
                message_id=approval_message_id,
                text="Working on the task.",
                parse_mode="HTML",
                reply_markup={"inline_keyboard": []},
            )
        else:
            await client.send_message(
                chat_id=chat_id,
                text="Working on the task.",
                parse_mode="HTML",
            )

    loader_state: dict[str, Any] = {"message_id": None}
    loader_stop = asyncio.Event()
    loader_task = asyncio.create_task(
        _animate_loader_until_done(
            client=client,
            chat_id=chat_id,
            state=loader_state,
            stop_event=loader_stop,
        )
    )
    try:
        outcome = await _execute_approved_action_and_summarize(
            get_agent_runtime=get_agent_runtime,
            get_context_events=get_context_events,
            approval_id=approval_id,
            decision_payload=decision_payload,
            chat_id=chat_id,
        )
    finally:
        if not loader_stop.is_set():
            loader_stop.set()
        with suppress(Exception):
            await loader_task

    message_id = loader_state.get("message_id")
    if isinstance(message_id, int):
        with suppress(Exception):
            await client.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=outcome,
                parse_mode="HTML",
            )
        return
    with suppress(Exception):
        await client.send_message(chat_id=chat_id, text=outcome, parse_mode="HTML")


def register_telegram_webhook_routes(
    app: FastAPI,
    *,
    settings: Any,
    get_telegram_client: Callable[[], Any],
    get_telegram_chat: Callable[[], Any],
    get_agent_runtime: Callable[[], Any],
    get_context_events: Callable[[], Any],
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
                enqueue_wake=False,
            )
            approved = bool(result.get("ok")) and str(result.get("status", "")).strip() == "approved"
            denied = bool(result.get("ok")) and str(result.get("status", "")).strip() == "denied"
            if approved:
                ack_text = "Working on the task."
            elif denied:
                ack_text = "Action denied."
            else:
                ack_text = f"Approval {decision} failed: {result.get('reason', 'not_allowed')}"
            with suppress(Exception):
                await get_telegram_client().answer_callback_query(
                    callback_query_id=callback_id,
                    text=ack_text,
                    show_alert=False,
                )
            if approved:
                await _run_post_approval_execution_flow(
                    get_telegram_client=get_telegram_client,
                    get_agent_runtime=get_agent_runtime,
                    get_context_events=get_context_events,
                    approval_id=approval_id,
                    decision_payload=result if isinstance(result, dict) else {},
                    chat_id=callback_chat_id,
                    approval_message_id=callback_message_id if isinstance(callback_message_id, int) else None,
                )
            elif isinstance(callback_message_id, int):
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
                enqueue_wake=False,
            )
            approved = bool(result.get("ok")) and str(result.get("status", "")).strip() == "approved"
            if approved:
                await _run_post_approval_execution_flow(
                    get_telegram_client=get_telegram_client,
                    get_agent_runtime=get_agent_runtime,
                    get_context_events=get_context_events,
                    approval_id=approval_id,
                    decision_payload=result if isinstance(result, dict) else {},
                    chat_id=int(chat_id),
                )
            else:
                reply_text = (
                    "Action denied."
                    if bool(result.get("ok")) and str(result.get("status", "")).strip() == "denied"
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
