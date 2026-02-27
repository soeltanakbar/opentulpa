"""API error helpers and request parsing utilities."""

from __future__ import annotations

from typing import TypeVar

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

TModel = TypeVar("TModel", bound=BaseModel)


def _format_validation_error(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "invalid request body"
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
    message = str(first.get("msg", "invalid value")).strip()
    if location:
        return f"{location}: {message}"
    return message or "invalid request body"


async def parse_request_model(
    request: Request,
    model: type[TModel],
) -> tuple[TModel | None, JSONResponse | None]:
    try:
        payload = await request.json()
    except Exception:
        return None, JSONResponse(status_code=400, content={"detail": "invalid JSON body"})
    try:
        return model.model_validate(payload), None
    except ValidationError as exc:
        return None, JSONResponse(
            status_code=400,
            content={"detail": _format_validation_error(exc)},
        )


def parse_query_model(
    request: Request,
    model: type[TModel],
) -> tuple[TModel | None, JSONResponse | None]:
    try:
        payload = dict(request.query_params)
        return model.model_validate(payload), None
    except ValidationError as exc:
        return None, JSONResponse(
            status_code=400,
            content={"detail": _format_validation_error(exc)},
        )
