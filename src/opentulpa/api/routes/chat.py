"""Internal conversation route registration (non-Telegram interface)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from opentulpa.api.errors import parse_request_model
from opentulpa.api.schemas.chat import InternalChatRequest
from opentulpa.domain.conversation import ConversationTurnRequest


def register_chat_routes(
    app: FastAPI,
    *,
    get_turn_orchestrator: Callable[[], Any],
) -> None:
    """Register API chat endpoints for direct (non-Telegram) turn simulation."""

    @app.post("/internal/chat")
    async def internal_chat(request: Request) -> Any:
        parsed, error = await parse_request_model(request, InternalChatRequest)
        if error is not None or parsed is None:
            return error
        customer_id = str(parsed.customer_id).strip()
        text = str(parsed.text).strip()
        thread_id = str(parsed.thread_id).strip()
        if not thread_id and customer_id:
            thread_id = f"chat-{customer_id}"
        include_pending_context = bool(parsed.include_pending_context)
        recursion_limit_override = parsed.recursion_limit_override

        if not customer_id or not text:
            return JSONResponse(
                status_code=400,
                content={"detail": "customer_id and text are required"},
            )
        orchestrator = get_turn_orchestrator()
        result = await orchestrator.run_turn(
            ConversationTurnRequest(
                customer_id=customer_id,
                thread_id=thread_id,
                text=text,
                include_pending_context=include_pending_context,
                recursion_limit_override=recursion_limit_override,
            )
        )
        status_code = 200 if result.status == "ok" else 503 if result.status == "unavailable" else 400
        return JSONResponse(
            status_code=status_code,
            content={
                "ok": result.status == "ok",
                "status": result.status,
                "customer_id": result.customer_id,
                "thread_id": result.thread_id,
                "text": result.text,
            },
        )
