"""Schemas for task routes."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from opentulpa.api.schemas.common import BaseRequestModel


class TaskCreateRequest(BaseRequestModel):
    customer_id: str = ""
    goal: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    risk_level: str = "low"
    idempotency_key: str | None = None


class TaskRelaunchRequest(BaseRequestModel):
    clarification: Any | None = None
    trigger_reason: str = "user_requested"


class TaskEventsQuery(BaseRequestModel):
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
