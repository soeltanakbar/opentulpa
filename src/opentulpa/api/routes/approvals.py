"""Approval-related internal API routes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from opentulpa.api.errors import parse_query_model, parse_request_model
from opentulpa.api.schemas.approvals import (
    ApprovalDecideRequest,
    ApprovalEvaluateRequest,
    ApprovalExecuteRequest,
    ApprovalPendingStatusQuery,
)
from opentulpa.application.approvals_orchestrator import (
    ApprovalsOrchestrator,
    ApprovalsOrchestratorResult,
)


def register_approval_routes(
    app: FastAPI,
    *,
    get_approvals: Callable[[], Any],
    get_wake_queue: Callable[[], Any],
    get_agent_runtime: Callable[[], Any],
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Register approval endpoints and return shared decision helper."""
    orchestrator = ApprovalsOrchestrator(
        get_approvals=get_approvals,
        get_wake_queue=get_wake_queue,
        get_agent_runtime=get_agent_runtime,
    )

    def _to_http_response(result: ApprovalsOrchestratorResult) -> Any:
        if result.status_code != 200:
            return JSONResponse(status_code=result.status_code, content=result.payload)
        return result.payload

    @app.post("/internal/approvals/evaluate")
    async def internal_approvals_evaluate(request: Request) -> Any:
        parsed, error = await parse_request_model(request, ApprovalEvaluateRequest)
        if error is not None or parsed is None:
            return error
        result = await orchestrator.evaluate_action(
            customer_id=parsed.customer_id,
            thread_id=parsed.thread_id,
            action_name=parsed.action_name,
            action_args=parsed.action_args if isinstance(parsed.action_args, dict) else {},
            action_note=parsed.action_note,
            guardrail_note=parsed.guardrail_note,
            origin_interface=parsed.origin_interface,
            origin_user_id=parsed.origin_user_id,
            origin_conversation_id=parsed.origin_conversation_id,
            defer_challenge_delivery=bool(parsed.defer_challenge_delivery),
        )
        return _to_http_response(result)

    @app.post("/internal/approvals/decide")
    async def internal_approvals_decide(request: Request) -> Any:
        parsed, error = await parse_request_model(request, ApprovalDecideRequest)
        if error is not None or parsed is None:
            return error
        result = await orchestrator.decide_action(
            approval_id=parsed.approval_id,
            decision=parsed.decision,
            actor_interface=parsed.actor_interface,
            actor_id=parsed.actor_id,
        )
        return _to_http_response(result)

    @app.post("/internal/approvals/execute")
    async def internal_approvals_execute(request: Request) -> Any:
        parsed, error = await parse_request_model(request, ApprovalExecuteRequest)
        if error is not None or parsed is None:
            return error
        result = await orchestrator.execute_action(
            approval_id=parsed.approval_id,
            customer_id=parsed.customer_id,
        )
        return _to_http_response(result)

    @app.get("/internal/approvals/{approval_id}")
    async def internal_approvals_get(approval_id: str) -> Any:
        result = orchestrator.get_approval(approval_id=approval_id)
        return _to_http_response(result)

    @app.get("/internal/approvals/pending/status")
    async def internal_approvals_pending_status(request: Request) -> Any:
        parsed, error = parse_query_model(request, ApprovalPendingStatusQuery)
        if error is not None or parsed is None:
            return error
        result = orchestrator.pending_status(
            customer_id=parsed.customer_id,
            thread_id=parsed.thread_id,
        )
        return _to_http_response(result)

    return orchestrator.decide_and_maybe_wake
