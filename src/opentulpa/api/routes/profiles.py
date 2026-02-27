"""Directive and time-profile route registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from opentulpa.api.errors import parse_request_model
from opentulpa.api.schemas.profiles import (
    DirectiveClearRequest,
    DirectiveGetRequest,
    DirectiveSetRequest,
    TimeProfileGetRequest,
    TimeProfileSetRequest,
)
from opentulpa.application.profile_orchestrator import (
    ProfileOrchestrator,
    ProfileOrchestratorResult,
)


def register_profile_routes(
    app: FastAPI,
    *,
    get_profiles: Callable[[], Any],
    get_memory: Callable[[], Any],
) -> None:
    """Register directive + timezone profile endpoints."""
    orchestrator = ProfileOrchestrator(
        get_profiles=get_profiles,
        get_memory=get_memory,
    )

    def _to_http_response(result: ProfileOrchestratorResult) -> Any:
        if result.status_code != 200:
            return JSONResponse(status_code=result.status_code, content=result.payload)
        return result.payload

    @app.post("/internal/directive/get")
    async def internal_directive_get(request: Request) -> Any:
        parsed, error = await parse_request_model(request, DirectiveGetRequest)
        if error is not None or parsed is None:
            return error
        return _to_http_response(orchestrator.get_directive(customer_id=parsed.customer_id))

    @app.post("/internal/directive/set")
    async def internal_directive_set(request: Request) -> Any:
        parsed, error = await parse_request_model(request, DirectiveSetRequest)
        if error is not None or parsed is None:
            return error
        return _to_http_response(
            orchestrator.set_directive(
                customer_id=parsed.customer_id,
                directive=parsed.directive,
                source=parsed.source,
            )
        )

    @app.post("/internal/directive/clear")
    async def internal_directive_clear(request: Request) -> Any:
        parsed, error = await parse_request_model(request, DirectiveClearRequest)
        if error is not None or parsed is None:
            return error
        return _to_http_response(orchestrator.clear_directive(customer_id=parsed.customer_id))

    @app.post("/internal/time_profile/get")
    async def internal_time_profile_get(request: Request) -> Any:
        parsed, error = await parse_request_model(request, TimeProfileGetRequest)
        if error is not None or parsed is None:
            return error
        return _to_http_response(orchestrator.get_time_profile(customer_id=parsed.customer_id))

    @app.post("/internal/time_profile/set")
    async def internal_time_profile_set(request: Request) -> Any:
        parsed, error = await parse_request_model(request, TimeProfileSetRequest)
        if error is not None or parsed is None:
            return error
        return _to_http_response(
            orchestrator.set_time_profile(
                customer_id=parsed.customer_id,
                utc_offset=parsed.utc_offset,
                source=parsed.source,
            )
        )
