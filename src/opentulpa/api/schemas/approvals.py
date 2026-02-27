"""Schemas for approval routes."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from opentulpa.api.schemas.common import BaseRequestModel


class ApprovalEvaluateRequest(BaseRequestModel):
    customer_id: str = ""
    thread_id: str = ""
    action_name: str = ""
    action_args: dict[str, Any] = Field(default_factory=dict)
    defer_challenge_delivery: bool = False
    action_note: str | None = None
    guardrail_note: str | None = None
    origin_interface: str | None = None
    origin_user_id: str | None = None
    origin_conversation_id: str | None = None


class ApprovalDecideRequest(BaseRequestModel):
    approval_id: str = ""
    decision: str = ""
    actor_interface: str = ""
    actor_id: str = ""


class ApprovalExecuteRequest(BaseRequestModel):
    approval_id: str = ""
    customer_id: str = ""


class ApprovalPendingStatusQuery(BaseRequestModel):
    customer_id: str = ""
    thread_id: str = ""
