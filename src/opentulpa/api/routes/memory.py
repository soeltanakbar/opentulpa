"""Internal memory route registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from opentulpa.api.errors import parse_request_model
from opentulpa.api.schemas.memory import MemoryAddRequest, MemorySearchRequest
from opentulpa.application.memory_orchestrator import MemoryOrchestrator, MemoryOrchestratorResult


def register_memory_routes(
    app: FastAPI,
    *,
    get_memory: Callable[[], Any],
) -> None:
    """Register internal memory add/search endpoints."""
    orchestrator = MemoryOrchestrator(get_memory=get_memory)

    def _to_http_response(result: MemoryOrchestratorResult) -> Any:
        if result.status_code != 200:
            return JSONResponse(status_code=result.status_code, content=result.payload)
        return result.payload

    @app.post("/internal/memory/add")
    async def internal_memory_add(request: Request) -> Any:
        parsed, error = await parse_request_model(request, MemoryAddRequest)
        if error is not None or parsed is None:
            return error
        result = orchestrator.add_memory(
            messages=parsed.messages,
            user_id=parsed.user_id,
            metadata=parsed.metadata,
            infer=parsed.infer,
            retries=parsed.retries,
        )
        return _to_http_response(result)

    @app.post("/internal/memory/search")
    async def internal_memory_search(request: Request) -> Any:
        parsed, error = await parse_request_model(request, MemorySearchRequest)
        if error is not None or parsed is None:
            return error
        result = orchestrator.search_memory(
            query=parsed.query,
            user_id=parsed.user_id,
            limit=parsed.limit,
        )
        return _to_http_response(result)
