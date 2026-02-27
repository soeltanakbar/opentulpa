"""Schemas for skill routes."""

from __future__ import annotations

from opentulpa.api.schemas.common import BaseRequestModel


class SkillListRequest(BaseRequestModel):
    customer_id: str = ""
    include_global: bool = True
    include_disabled: bool = False
    limit: int = 200


class SkillGetRequest(BaseRequestModel):
    customer_id: str = ""
    name: str = ""
    include_files: bool = True
    include_global: bool = True


class SkillUpsertRequest(BaseRequestModel):
    customer_id: str = ""
    scope: str = "user"
    name: str = ""
    description: str = ""
    instructions: str = ""
    skill_markdown: str = ""
    source: str = "agent"
    supporting_files: dict[str, str] | None = None


class SkillDeleteRequest(BaseRequestModel):
    customer_id: str = ""
    scope: str = "user"
    name: str = ""
