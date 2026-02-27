"""Skill store route registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from opentulpa.api.errors import parse_request_model
from opentulpa.api.schemas.skills import (
    SkillDeleteRequest,
    SkillGetRequest,
    SkillListRequest,
    SkillUpsertRequest,
)
from opentulpa.application.skill_orchestrator import (
    SkillOrchestrator,
    SkillOrchestratorResult,
)


def register_skill_routes(
    app: FastAPI,
    *,
    get_skill_store: Callable[[], Any],
    get_memory: Callable[[], Any],
) -> None:
    """Register internal skill list/get/upsert/delete endpoints."""
    orchestrator = SkillOrchestrator(
        get_skill_store=get_skill_store,
        get_memory=get_memory,
    )

    def _to_http_response(result: SkillOrchestratorResult) -> Any:
        if result.status_code != 200:
            return JSONResponse(status_code=result.status_code, content=result.payload)
        return result.payload

    @app.post("/internal/skills/list")
    async def internal_skills_list(request: Request) -> Any:
        parsed, error = await parse_request_model(request, SkillListRequest)
        if error is not None or parsed is None:
            return error
        result = orchestrator.list_skills(
            customer_id=str(parsed.customer_id).strip(),
            include_global=bool(parsed.include_global),
            include_disabled=bool(parsed.include_disabled),
            limit=int(parsed.limit),
        )
        return _to_http_response(result)

    @app.post("/internal/skills/get")
    async def internal_skills_get(request: Request) -> Any:
        parsed, error = await parse_request_model(request, SkillGetRequest)
        if error is not None or parsed is None:
            return error
        result = orchestrator.get_skill(
            customer_id=str(parsed.customer_id).strip(),
            name=str(parsed.name).strip(),
            include_files=bool(parsed.include_files),
            include_global=bool(parsed.include_global),
        )
        return _to_http_response(result)

    @app.post("/internal/skills/upsert")
    async def internal_skills_upsert(request: Request) -> Any:
        parsed, error = await parse_request_model(request, SkillUpsertRequest)
        if error is not None or parsed is None:
            return error
        result = orchestrator.upsert_skill(
            customer_id=str(parsed.customer_id).strip(),
            scope=str(parsed.scope).strip().lower(),
            name=str(parsed.name).strip(),
            description=str(parsed.description).strip(),
            instructions=str(parsed.instructions).strip(),
            skill_markdown=str(parsed.skill_markdown).strip(),
            source=str(parsed.source or "agent"),
            supporting_files=(
                parsed.supporting_files if isinstance(parsed.supporting_files, dict) else None
            ),
        )
        return _to_http_response(result)

    @app.post("/internal/skills/delete")
    async def internal_skills_delete(request: Request) -> Any:
        parsed, error = await parse_request_model(request, SkillDeleteRequest)
        if error is not None or parsed is None:
            return error
        result = orchestrator.delete_skill(
            customer_id=str(parsed.customer_id).strip(),
            scope=str(parsed.scope).strip().lower(),
            name=str(parsed.name).strip(),
        )
        return _to_http_response(result)
