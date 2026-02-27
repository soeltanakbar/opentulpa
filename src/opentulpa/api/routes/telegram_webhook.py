"""Telegram webhook route registration."""

from __future__ import annotations

import hmac
import re
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import JSONResponse

from opentulpa.api.errors import parse_request_model
from opentulpa.api.schemas.telegram import TelegramWebhookRequest
from opentulpa.application.telegram_webhook_orchestrator import (
    TelegramWebhookOrchestrator,
)


def _parse_approval_callback_data(data: str) -> tuple[str, str] | None:
    raw = str(data or "").strip()
    # Format: approval:<approval_id>:approve|deny
    match = re.fullmatch(
        r"approval:([a-z0-9_-]{6,40}):(approve|deny)",
        raw,
        re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1), match.group(2).lower()

def register_telegram_webhook_routes(
    app: FastAPI,
    *,
    settings: Any,
    get_telegram_client: Callable[[], Any],
    get_telegram_chat: Callable[[], Any],
    get_approvals: Callable[[], Any],
    get_agent_runtime: Callable[[], Any],
    get_approval_execution_orchestrator: Callable[[], Any],
    decide_approval_and_maybe_wake: Callable[..., Awaitable[dict[str, Any]]],
) -> None:
    """Register Telegram webhook with callback-driven approval support."""
    orchestrator = TelegramWebhookOrchestrator(
        settings=settings,
        get_telegram_client=get_telegram_client,
        get_telegram_chat=get_telegram_chat,
        get_approvals=get_approvals,
        get_agent_runtime=get_agent_runtime,
        get_approval_execution_orchestrator=get_approval_execution_orchestrator,
        decide_approval_and_maybe_wake=decide_approval_and_maybe_wake,
        parse_approval_callback_data=_parse_approval_callback_data,
    )

    @app.post("/webhook/telegram")
    async def telegram_webhook(request: Request, background_tasks: BackgroundTasks) -> Response:
        if not settings.telegram_bot_token:
            return JSONResponse(status_code=501, content={"detail": "Telegram not configured"})
        expected_secret = str(settings.telegram_webhook_secret or "").strip()
        if not expected_secret:
            return JSONResponse(
                status_code=503,
                content={"detail": "telegram webhook secret not configured"},
            )
        incoming_secret = str(request.headers.get("x-telegram-bot-api-secret-token", "") or "").strip()
        if not hmac.compare_digest(incoming_secret, expected_secret):
            return JSONResponse(status_code=403, content={"detail": "invalid telegram secret"})
        parsed, error = await parse_request_model(request, TelegramWebhookRequest)
        if error is not None or parsed is None:
            return error
        body = parsed.model_dump(exclude_none=True)

        # Immediate 200 OK, logic runs in background.
        background_tasks.add_task(orchestrator.handle_background_update, body=body)
        return Response(status_code=200)
