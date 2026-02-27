"""Scheduler route registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from opentulpa.api.errors import parse_query_model, parse_request_model
from opentulpa.api.schemas.scheduler import (
    RoutineCreateRequest,
    RoutineDeleteWithAssetsRequest,
    SchedulerRoutineDeleteQuery,
    SchedulerRoutinesQuery,
)
from opentulpa.application.scheduler_orchestrator import (
    SchedulerOrchestrator,
    SchedulerOrchestratorResult,
)


def register_scheduler_routes(
    app: FastAPI,
    *,
    get_scheduler: Callable[[], Any],
    delete_file: Callable[..., dict[str, Any]],
) -> None:
    """Register scheduler routine management endpoints."""
    orchestrator = SchedulerOrchestrator(
        get_scheduler=get_scheduler,
        delete_file=delete_file,
    )

    def _to_http_response(result: SchedulerOrchestratorResult) -> Any:
        if result.status_code != 200:
            return JSONResponse(status_code=result.status_code, content=result.payload)
        return result.payload

    @app.post("/internal/scheduler/routine")
    async def internal_scheduler_add_routine(request: Request) -> Any:
        parsed, error = await parse_request_model(request, RoutineCreateRequest)
        if error is not None or parsed is None:
            return error
        result = orchestrator.create_routine(
            routine_id=str(parsed.id).strip(),
            name=parsed.name,
            schedule=parsed.schedule,
            payload=parsed.payload if isinstance(parsed.payload, dict) else {},
            enabled=parsed.enabled,
            is_cron=parsed.is_cron,
        )
        return _to_http_response(result)

    @app.get("/internal/scheduler/routines")
    async def internal_scheduler_list_routines(request: Request) -> Any:
        parsed, error = parse_query_model(request, SchedulerRoutinesQuery)
        if error is not None or parsed is None:
            return error
        result = orchestrator.list_routines(customer_id=parsed.customer_id)
        return _to_http_response(result)

    @app.delete("/internal/scheduler/routine/{routine_id}")
    async def internal_scheduler_remove_routine(
        routine_id: str,
        request: Request,
    ) -> Any:
        parsed, error = parse_query_model(request, SchedulerRoutineDeleteQuery)
        if error is not None or parsed is None:
            return error
        result = orchestrator.remove_routine(
            routine_id=routine_id,
            customer_id=parsed.customer_id,
        )
        return _to_http_response(result)

    @app.post("/internal/scheduler/routine/delete_with_assets")
    async def internal_scheduler_remove_routine_with_assets(request: Request) -> Any:
        parsed, error = await parse_request_model(request, RoutineDeleteWithAssetsRequest)
        if error is not None or parsed is None:
            return error
        result = orchestrator.remove_routine_with_assets(
            customer_id=str(parsed.customer_id).strip(),
            routine_id=str(parsed.routine_id).strip(),
            name=str(parsed.name).strip(),
            remove_all_matches=bool(parsed.remove_all_matches),
            delete_files=bool(parsed.delete_files),
            cleanup_paths=parsed.cleanup_paths,
        )
        return _to_http_response(result)
