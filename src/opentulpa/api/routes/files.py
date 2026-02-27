"""Internal file-vault route registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from opentulpa.api.errors import parse_request_model
from opentulpa.api.schemas.files import (
    FileAnalyzeRequest,
    FileGetRequest,
    FileSearchRequest,
    FileSendLocalRequest,
    FileSendRequest,
    FileSendWebImageRequest,
)
from opentulpa.application.file_orchestrator import FileOrchestrator, FileOrchestratorResult


def register_file_routes(
    app: FastAPI,
    *,
    get_file_vault: Callable[[], Any],
    get_telegram_chat: Callable[[], Any],
    get_telegram_client: Callable[[], Any],
    get_agent_runtime: Callable[[], Any],
    telegram_enabled: bool,
) -> None:
    """Register uploaded-file search/get/send/analyze endpoints."""
    orchestrator = FileOrchestrator(
        get_file_vault=get_file_vault,
        get_telegram_chat=get_telegram_chat,
        get_telegram_client=get_telegram_client,
        get_agent_runtime=get_agent_runtime,
        telegram_enabled=telegram_enabled,
    )

    def _to_http_response(result: FileOrchestratorResult) -> Any:
        if result.status_code != 200:
            return JSONResponse(status_code=result.status_code, content=result.payload)
        return result.payload

    @app.post("/internal/files/search")
    async def internal_files_search(request: Request) -> Any:
        parsed, error = await parse_request_model(request, FileSearchRequest)
        if error is not None or parsed is None:
            return error
        result = orchestrator.search_files(
            customer_id=str(parsed.customer_id).strip(),
            query=str(parsed.query).strip(),
            limit=int(parsed.limit),
        )
        return _to_http_response(result)

    @app.post("/internal/files/get")
    async def internal_files_get(request: Request) -> Any:
        parsed, error = await parse_request_model(request, FileGetRequest)
        if error is not None or parsed is None:
            return error
        result = orchestrator.get_file(
            customer_id=str(parsed.customer_id).strip(),
            file_id=str(parsed.file_id).strip(),
            max_excerpt_chars=max(500, min(int(parsed.max_excerpt_chars), 60000)),
        )
        return _to_http_response(result)

    @app.post("/internal/files/send")
    async def internal_files_send(request: Request) -> Any:
        parsed, error = await parse_request_model(request, FileSendRequest)
        if error is not None or parsed is None:
            return error
        caption = str(parsed.caption).strip() if parsed.caption is not None else None
        result = await orchestrator.send_file(
            customer_id=str(parsed.customer_id).strip(),
            file_id=str(parsed.file_id).strip(),
            caption=caption or None,
        )
        return _to_http_response(result)

    @app.post("/internal/files/send_local")
    async def internal_files_send_local(request: Request) -> Any:
        parsed, error = await parse_request_model(request, FileSendLocalRequest)
        if error is not None or parsed is None:
            return error
        caption = str(parsed.caption).strip() if parsed.caption is not None else None
        result = await orchestrator.send_local_file(
            customer_id=str(parsed.customer_id).strip(),
            local_path=str(parsed.path).strip(),
            caption=caption or None,
        )
        return _to_http_response(result)

    @app.post("/internal/files/send_web_image")
    async def internal_files_send_web_image(request: Request) -> Any:
        parsed, error = await parse_request_model(request, FileSendWebImageRequest)
        if error is not None or parsed is None:
            return error
        caption = str(parsed.caption).strip() if parsed.caption is not None else None
        result = await orchestrator.send_web_image(
            customer_id=str(parsed.customer_id).strip(),
            image_url=str(parsed.url).strip(),
            caption=caption or None,
            max_bytes=int(parsed.max_bytes),
        )
        return _to_http_response(result)

    @app.post("/internal/files/analyze")
    async def internal_files_analyze(request: Request) -> Any:
        parsed, error = await parse_request_model(request, FileAnalyzeRequest)
        if error is not None or parsed is None:
            return error
        question = str(parsed.question).strip() if parsed.question is not None else None
        result = await orchestrator.analyze_file(
            customer_id=str(parsed.customer_id).strip(),
            file_id=str(parsed.file_id).strip(),
            question=question or None,
        )
        return _to_http_response(result)
