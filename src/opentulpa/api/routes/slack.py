"""Slack internal API route registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from opentulpa.api.errors import parse_query_model, parse_request_model
from opentulpa.api.schemas.slack import (
    SlackChannelHistoryQuery,
    SlackChannelsQuery,
    SlackConsentRequest,
    SlackPostRequest,
)
from opentulpa.application.slack_orchestrator import SlackOrchestrator, SlackOrchestratorResult


def register_slack_routes(
    app: FastAPI,
    *,
    get_slack: Callable[[], Any],
    has_write_consent: Callable[[str], bool],
    grant_write_consent: Callable[[str], None],
) -> None:
    """Register internal Slack endpoints used by tools."""
    orchestrator = SlackOrchestrator(
        get_slack=get_slack,
        has_write_consent=has_write_consent,
        grant_write_consent=grant_write_consent,
    )

    def _to_http_response(result: SlackOrchestratorResult) -> Any:
        if result.status_code != 200:
            return JSONResponse(status_code=result.status_code, content=result.payload)
        return result.payload

    @app.get("/internal/slack/channels")
    async def internal_slack_channels(request: Request) -> Any:
        parsed, error = parse_query_model(request, SlackChannelsQuery)
        if error is not None or parsed is None:
            return error
        result = await orchestrator.list_channels(limit=parsed.limit, cursor=parsed.cursor)
        return _to_http_response(result)

    @app.get("/internal/slack/channels/{channel_id}/history")
    async def internal_slack_history(channel_id: str, request: Request) -> Any:
        parsed, error = parse_query_model(request, SlackChannelHistoryQuery)
        if error is not None or parsed is None:
            return error
        result = await orchestrator.channel_history(
            channel_id=channel_id,
            limit=parsed.limit,
            cursor=parsed.cursor,
        )
        return _to_http_response(result)

    @app.post("/internal/slack/consent")
    async def internal_slack_consent(request: Request) -> Any:
        """Grant Slack write consent for this customer (called when user confirms in chat)."""
        parsed, error = await parse_request_model(request, SlackConsentRequest)
        if error is not None or parsed is None:
            return error
        result = orchestrator.grant_consent(customer_id=parsed.customer_id, scope=parsed.scope)
        return _to_http_response(result)

    @app.post("/internal/slack/post")
    async def internal_slack_post(request: Request) -> Any:
        parsed, error = await parse_request_model(request, SlackPostRequest)
        if error is not None or parsed is None:
            return error
        result = await orchestrator.post_message(
            customer_id=parsed.customer_id,
            channel_id=parsed.channel_id,
            text=parsed.text,
            thread_ts=parsed.thread_ts,
        )
        return _to_http_response(result)
