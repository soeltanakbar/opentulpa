"""LLM-driven policy evaluation for side-effect guardrails."""

from __future__ import annotations

from typing import Any

__all__ = ["ApprovalEvaluator", "ExecutionBoundaryContext", "ExecutionBoundaryGuard"]


def __getattr__(name: str) -> Any:
    if name == "ApprovalEvaluator":
        from opentulpa.policy.evaluator import ApprovalEvaluator

        return ApprovalEvaluator
    if name in {"ExecutionBoundaryContext", "ExecutionBoundaryGuard"}:
        from opentulpa.policy.execution_boundary import (
            ExecutionBoundaryContext,
            ExecutionBoundaryGuard,
        )

        if name == "ExecutionBoundaryContext":
            return ExecutionBoundaryContext
        return ExecutionBoundaryGuard
    raise AttributeError(name)
