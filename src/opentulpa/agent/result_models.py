"""Typed runtime result contracts for agent/policy boundaries."""

from __future__ import annotations

from contextlib import suppress
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field


class AgentResultModel(BaseModel):
    """Base typed result model."""

    model_config = ConfigDict(extra="allow")


class WakeEventDecision(AgentResultModel):
    notify_user: bool = False
    reason: str = ""

    @classmethod
    def from_any(cls, payload: Any) -> Self:
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, dict):
            with suppress(Exception):
                return cls.model_validate(payload)
        return cls(notify_user=False, reason="invalid_wake_decision_payload")


class CompletionClaimVerification(AgentResultModel):
    ok: bool = False
    applies: bool = False
    mismatch: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    repair_instruction: str = ""
    usable: bool = False

    @classmethod
    def from_any(cls, payload: Any) -> Self:
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, dict):
            with suppress(Exception):
                return cls.model_validate(payload)
        return cls(
            ok=False,
            applies=False,
            mismatch=False,
            confidence=0.0,
            reason="invalid_completion_claim_payload",
            repair_instruction="",
            usable=False,
        )


class GuardrailIntentDecision(AgentResultModel):
    ok: bool = False
    gate: Literal["allow", "require_approval", "deny"] | None = None
    impact_type: Literal["read", "write", "purchase", "costly"] | None = None
    recipient_scope: Literal["self", "external", "unknown"] | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    error: str = ""

    @classmethod
    def from_any(cls, payload: Any) -> Self:
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, dict):
            with suppress(Exception):
                return cls.model_validate(payload)
        return cls(ok=False, error="invalid_guardrail_intent_payload")


class ToolGuardrailDecision(AgentResultModel):
    gate: Literal["allow", "require_approval", "deny"] = "require_approval"
    reason: str = "approval_required"
    summary: str = ""
    impact_type: str | None = None
    recipient_scope: str | None = None
    confidence: float | None = None
    approval_id: str | None = None
    status: str | None = None
    expires_at: str | None = None
    delivery_mode: str | None = None

    @classmethod
    def from_any(
        cls,
        payload: Any,
        *,
        default_summary: str = "",
        default_reason: str = "approval_required",
    ) -> Self:
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, dict):
            with suppress(Exception):
                decision = cls.model_validate(payload)
                if not decision.summary and default_summary:
                    return decision.model_copy(update={"summary": default_summary})
                return decision
        return cls(summary=default_summary, reason=default_reason)
