"""Schemas for scheduler routes."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from opentulpa.api.schemas.common import BaseRequestModel


class RoutineCreateRequest(BaseRequestModel):
    id: str = ""
    name: str = "Unnamed"
    schedule: str = "0 9 * * *"
    payload: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    is_cron: bool = True


class RoutineDeleteWithAssetsRequest(BaseRequestModel):
    customer_id: str = ""
    routine_id: str = ""
    name: str = ""
    remove_all_matches: bool = False
    delete_files: bool = True
    cleanup_paths: list[str] = Field(default_factory=list)


class SchedulerRoutinesQuery(BaseRequestModel):
    customer_id: str | None = None


class SchedulerRoutineDeleteQuery(BaseRequestModel):
    customer_id: str | None = None
