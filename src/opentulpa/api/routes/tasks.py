"""Task orchestration route registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from opentulpa.api.errors import parse_query_model, parse_request_model
from opentulpa.api.schemas.tasks import TaskCreateRequest, TaskEventsQuery, TaskRelaunchRequest
from opentulpa.application.task_orchestrator import TaskOrchestrator, TaskOrchestratorResult


def register_task_routes(
    app: FastAPI,
    *,
    get_tasks: Callable[[], Any],
) -> None:
    """Register task create/status/event/artifact/control endpoints."""
    orchestrator = TaskOrchestrator(get_tasks=get_tasks)

    def _to_http_response(result: TaskOrchestratorResult) -> Any:
        if result.status_code != 200:
            return JSONResponse(status_code=result.status_code, content=result.payload)
        return result.payload

    @app.post("/internal/tasks/create")
    async def internal_task_create(request: Request) -> Any:
        parsed, error = await parse_request_model(request, TaskCreateRequest)
        if error is not None or parsed is None:
            return error
        result = await orchestrator.create_task(
            customer_id=parsed.customer_id,
            goal=parsed.goal,
            payload=parsed.payload if isinstance(parsed.payload, dict) else {},
            risk_level=parsed.risk_level,
            idempotency_key=parsed.idempotency_key,
        )
        return _to_http_response(result)

    @app.get("/internal/tasks/{task_id}")
    async def internal_task_status(task_id: str) -> Any:
        result = orchestrator.get_task(task_id=task_id)
        return _to_http_response(result)

    @app.get("/internal/tasks/{task_id}/events")
    async def internal_task_events(task_id: str, request: Request) -> Any:
        parsed, error = parse_query_model(request, TaskEventsQuery)
        if error is not None or parsed is None:
            return error
        result = orchestrator.list_events(task_id=task_id, limit=parsed.limit, offset=parsed.offset)
        return _to_http_response(result)

    @app.get("/internal/tasks/{task_id}/artifacts")
    async def internal_task_artifacts(task_id: str) -> Any:
        result = orchestrator.list_artifacts(task_id=task_id)
        return _to_http_response(result)

    @app.post("/internal/tasks/{task_id}/relaunch")
    async def internal_task_relaunch(task_id: str, request: Request) -> Any:
        parsed, error = await parse_request_model(request, TaskRelaunchRequest)
        if error is not None or parsed is None:
            return error
        result = await orchestrator.relaunch_task(
            task_id=task_id,
            clarification=parsed.clarification,
            trigger_reason=parsed.trigger_reason,
        )
        return _to_http_response(result)

    @app.post("/internal/tasks/{task_id}/cancel")
    async def internal_task_cancel(task_id: str) -> Any:
        result = await orchestrator.cancel_task(task_id=task_id)
        return _to_http_response(result)
