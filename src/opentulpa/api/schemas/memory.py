"""Schemas for memory routes."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from opentulpa.api.schemas.common import BaseRequestModel


class MemoryAddRequest(BaseRequestModel):
    messages: list[dict[str, Any]] = Field(default_factory=list)
    user_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    infer: bool = True
    retries: int = 1


class MemorySearchRequest(BaseRequestModel):
    query: str = ""
    user_id: str | None = None
    limit: int = 5
