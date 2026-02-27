"""Execution-boundary guard orchestration for executable actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opentulpa.agent.result_models import ToolGuardrailDecision

_SCHEDULED_ORIGINS = {"scheduled", "schedule", "routine", "wake", "background"}


@dataclass(slots=True)
class ExecutionBoundaryContext:
    customer_id: str
    thread_id: str
    action_name: str
    action_args: dict[str, Any]
    execution_origin: str | None = None
    preapproved: bool = False
    action_note: str | None = None


class ExecutionBoundaryGuard:
    """
    Centralized execution-boundary guard.

    Guardrails are evaluated only for interactive executable actions.
    Scheduled/wake-origin executions are pre-authorized at schedule creation time.
    """

    def __init__(self, *, runtime: Any) -> None:
        self._runtime = runtime

    @staticmethod
    def _normalize_thread_id(thread_id: str | None) -> str:
        return str(thread_id or "").strip()

    @classmethod
    def normalize_execution_origin(cls, *, thread_id: str | None, execution_origin: str | None) -> str:
        raw_origin = str(execution_origin or "").strip().lower()
        if raw_origin in _SCHEDULED_ORIGINS:
            return "scheduled"
        if raw_origin in {"interactive", "manual", "chat"}:
            return "interactive"
        safe_thread = cls._normalize_thread_id(thread_id).lower()
        if safe_thread.startswith("wake_") or safe_thread.startswith("wake-"):
            return "scheduled"
        return "interactive"

    async def evaluate(self, context: ExecutionBoundaryContext) -> ToolGuardrailDecision:
        if bool(context.preapproved):
            return ToolGuardrailDecision(
                gate="allow",
                reason="approval_execution_preapproved",
                summary=f"execute {context.action_name}",
            )

        normalized_origin = self.normalize_execution_origin(
            thread_id=context.thread_id,
            execution_origin=context.execution_origin,
        )
        if normalized_origin == "scheduled":
            return ToolGuardrailDecision(
                gate="allow",
                reason="scheduled_execution_skip_guardrail",
                summary=f"execute {context.action_name}",
            )

        if not hasattr(self._runtime, "evaluate_tool_guardrail"):
            return ToolGuardrailDecision(
                gate="allow",
                reason="guardrail_runtime_unavailable_allow",
                summary=f"execute {context.action_name}",
            )

        try:
            payload = await self._runtime.evaluate_tool_guardrail(
                customer_id=str(context.customer_id or "").strip(),
                thread_id=self._normalize_thread_id(context.thread_id) or "interactive",
                action_name=str(context.action_name or "").strip(),
                action_args=context.action_args if isinstance(context.action_args, dict) else {},
                action_note=str(context.action_note or "").strip()[:2000] or None,
            )
            return ToolGuardrailDecision.from_any(
                payload,
                default_summary=f"execute {context.action_name}",
                default_reason="guardrail_boundary_invalid_payload",
            )
        except Exception as exc:
            return ToolGuardrailDecision(
                gate="require_approval",
                reason=f"guardrail_boundary_error:{exc}",
                summary=f"execute {context.action_name}",
            )
