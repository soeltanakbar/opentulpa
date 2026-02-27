"""Core data models for external-impact approval guardrails."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

RecipientScope = Literal["self", "external", "unknown"]
ImpactType = Literal["read", "write", "purchase", "costly"]
ApprovalStatus = Literal["pending", "approved", "denied", "expired", "executed"]
GateAction = Literal["allow", "require_approval", "deny"]


@dataclass(slots=True)
class ActionIntent:
    customer_id: str
    thread_id: str
    action_name: str
    action_args: dict[str, Any]
    origin_interface: str
    origin_user_id: str
    origin_conversation_id: str
    recipient_scope: RecipientScope
    impact_type: ImpactType
    summary: str
    reason: str
    confidence: float
    llm_gate: GateAction | None = None
    llm_uncertain: bool = False


@dataclass(slots=True)
class ApprovalRecord:
    id: str
    customer_id: str
    thread_id: str
    origin_interface: str
    origin_user_id: str
    origin_conversation_id: str
    action_name: str
    action_args_json: str
    recipient_scope: RecipientScope
    impact_type: ImpactType
    summary: str
    reason: str
    confidence: float
    status: ApprovalStatus
    created_at: str
    expires_at: str
    decided_at: str | None
    executed_at: str | None
    decision_actor_id: str | None


@dataclass(slots=True)
class GateDecision:
    gate: GateAction
    reason: str
    summary: str
    confidence: float
    recipient_scope: RecipientScope
    impact_type: ImpactType
    approval_id: str | None = None
    status: ApprovalStatus | None = None
    expires_at: str | None = None
    delivery_mode: str | None = None
