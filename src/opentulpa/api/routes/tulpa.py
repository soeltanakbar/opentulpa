"""Dynamic tulpa sandbox route registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from opentulpa.api.errors import parse_query_model, parse_request_model
from opentulpa.api.schemas.tulpa import (
    TulpaReadFileQuery,
    TulpaRunTerminalRequest,
    TulpaValidateFileRequest,
    TulpaWriteFileRequest,
)
from opentulpa.application.tulpa_orchestrator import TulpaOrchestrator, TulpaOrchestratorResult


def register_tulpa_routes(
    app: FastAPI,
    *,
    get_tulpa_loader: Callable[[], Any],
) -> None:
    """Register tulpa reload/read/write/run validation endpoints."""
    orchestrator = TulpaOrchestrator(get_tulpa_loader=get_tulpa_loader)

    def _to_http_response(result: TulpaOrchestratorResult) -> Any:
        if result.status_code != 200:
            return JSONResponse(status_code=result.status_code, content=result.payload)
        return result.payload

    @app.post("/internal/tulpa/reload")
    async def internal_tulpa_reload() -> Any:
        """Reload APIRouter modules from tulpa_stuff."""
        return _to_http_response(orchestrator.reload_modules())

    @app.post("/internal/tulpa/write_file")
    async def internal_tulpa_write_file(request: Request) -> Any:
        """Write a file only inside approved integration/self-modification paths."""
        parsed, error = await parse_request_model(request, TulpaWriteFileRequest)
        if error is not None or parsed is None:
            return error
        result = orchestrator.write_file(
            relative_path=str(parsed.path).strip(),
            content=parsed.content,
        )
        return _to_http_response(result)

    @app.post("/internal/tulpa/validate_file")
    async def internal_tulpa_validate_file(request: Request) -> Any:
        """Validate generated code file contract/syntax before using it."""
        parsed, error = await parse_request_model(request, TulpaValidateFileRequest)
        if error is not None or parsed is None:
            return error
        result = orchestrator.validate_file(relative_path=str(parsed.path).strip())
        return _to_http_response(result)

    @app.get("/internal/tulpa/read_file")
    async def internal_tulpa_read_file(request: Request) -> Any:
        """Read a file inside approved integration/self-modification paths."""
        parsed, error = parse_query_model(request, TulpaReadFileQuery)
        if error is not None or parsed is None:
            return error
        result = orchestrator.read_file(
            path=str(parsed.path).strip(),
            max_chars=int(parsed.max_chars),
        )
        return _to_http_response(result)

    @app.post("/internal/tulpa/run_terminal")
    async def internal_tulpa_run_terminal(request: Request) -> Any:
        """Run a restricted command in approved integration/self-modification paths."""
        parsed, error = await parse_request_model(request, TulpaRunTerminalRequest)
        if error is not None or parsed is None:
            return error
        command = str(parsed.command).strip()
        result = orchestrator.run_terminal(
            command=command,
            working_dir_key=str(parsed.working_dir).strip(),
            timeout_seconds=int(parsed.timeout_seconds),
        )
        return _to_http_response(result)

    @app.get("/internal/tulpa/catalog")
    async def internal_tulpa_catalog() -> Any:
        """Return tulpa_stuff catalog/index and recent tracked entries."""
        return _to_http_response(orchestrator.catalog())
