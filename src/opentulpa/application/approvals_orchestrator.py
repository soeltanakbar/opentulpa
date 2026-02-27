"""Application-layer orchestration for approval APIs."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from opentulpa.application.contracts import ApplicationResult

logger = logging.getLogger(__name__)


class ApprovalsOrchestratorResult(ApplicationResult[dict[str, object]]):
    """Normalized route-friendly result payload."""


class ApprovalsOrchestrator:
    """Owns approval endpoint business rules independent of FastAPI transport."""

    def __init__(
        self,
        *,
        get_approvals: Callable[[], Any],
        get_wake_queue: Callable[[], Any],
        get_agent_runtime: Callable[[], Any],
    ) -> None:
        self._get_approvals = get_approvals
        self._get_wake_queue = get_wake_queue
        self._get_agent_runtime = get_agent_runtime

    async def decide_and_maybe_wake(
        self,
        *,
        approval_id: str,
        decision: str,
        actor_interface: str,
        actor_id: str,
        enqueue_wake: bool = True,
    ) -> dict[str, object]:
        broker = self._get_approvals()
        resolved = await broker.decide(
            approval_id=approval_id,
            decision=decision,
            actor_interface=actor_interface,
            actor_id=actor_id,
        )
        resolved_status = str(resolved.get("status", "")).strip()
        if enqueue_wake and bool(resolved.get("ok")) and resolved_status in {"approved", "denied"}:
            approval_ref = str(resolved.get("id", approval_id)).strip()
            payload = {
                "type": "approval_event",
                "event_type": resolved_status,
                "customer_id": str(resolved.get("customer_id", "")).strip(),
                "thread_id": str(resolved.get("thread_id", "")).strip(),
                "approval_id": approval_ref,
                "payload": {
                    "approval_id": approval_ref,
                    "action_name": str(resolved.get("action_name", "")).strip(),
                    "summary": str(resolved.get("summary", "")).strip(),
                    "status": resolved_status,
                    "reason": str(resolved.get("reason", "")).strip(),
                },
            }
            try:
                queue_id = await self._get_wake_queue().enqueue(payload)
                resolved["wake_queued"] = True
                resolved["wake_queue_id"] = queue_id
            except Exception:
                resolved["wake_queued"] = False
        return resolved

    async def evaluate_action(
        self,
        *,
        customer_id: str,
        thread_id: str,
        action_name: str,
        action_args: dict[str, object],
        defer_challenge_delivery: bool,
        action_note: str | None = None,
        guardrail_note: str | None = None,
        origin_interface: str | None = None,
        origin_user_id: str | None = None,
        origin_conversation_id: str | None = None,
    ) -> ApprovalsOrchestratorResult:
        cid = str(customer_id or "").strip()
        tid = str(thread_id or "").strip()
        safe_action_name = str(action_name or "").strip()
        if not cid or not tid or not safe_action_name:
            return ApprovalsOrchestratorResult(
                status_code=400,
                payload={"detail": "customer_id, thread_id, and action_name are required"},
            )
        normalized_action_note = str(action_note or "").strip() or str(guardrail_note or "").strip() or None
        normalized_origin_interface = str(origin_interface or "").strip() or None
        normalized_origin_user_id = str(origin_user_id or "").strip() or None
        normalized_origin_conversation_id = str(origin_conversation_id or "").strip() or None
        broker = self._get_approvals()
        decision = await broker.evaluate_action(
            customer_id=cid,
            thread_id=tid,
            action_name=safe_action_name,
            action_args=action_args if isinstance(action_args, dict) else {},
            action_note=normalized_action_note,
            origin_interface=normalized_origin_interface,
            origin_user_id=normalized_origin_user_id,
            origin_conversation_id=normalized_origin_conversation_id,
            defer_challenge_delivery=bool(defer_challenge_delivery),
        )
        return ApprovalsOrchestratorResult(status_code=200, payload=decision)

    async def decide_action(
        self,
        *,
        approval_id: str,
        decision: str,
        actor_interface: str,
        actor_id: str,
    ) -> ApprovalsOrchestratorResult:
        safe_approval_id = str(approval_id or "").strip()
        safe_decision = str(decision or "").strip().lower()
        safe_actor_interface = str(actor_interface or "").strip()
        safe_actor_id = str(actor_id or "").strip()
        if (
            not safe_approval_id
            or safe_decision not in {"approve", "deny"}
            or not safe_actor_interface
            or not safe_actor_id
        ):
            return ApprovalsOrchestratorResult(
                status_code=400,
                payload={
                    "detail": (
                        "approval_id, decision (approve|deny), actor_interface, and actor_id are required"
                    )
                },
            )
        resolved = await self.decide_and_maybe_wake(
            approval_id=safe_approval_id,
            decision=safe_decision,
            actor_interface=safe_actor_interface,
            actor_id=safe_actor_id,
        )
        return ApprovalsOrchestratorResult(status_code=200, payload=resolved)

    async def execute_action(
        self,
        *,
        approval_id: str,
        customer_id: str,
    ) -> ApprovalsOrchestratorResult:
        safe_approval_id = str(approval_id or "").strip()
        safe_customer_id = str(customer_id or "").strip()
        if not safe_approval_id or not safe_customer_id:
            return ApprovalsOrchestratorResult(
                status_code=400,
                payload={"detail": "approval_id and customer_id are required"},
            )
        broker = self._get_approvals()
        agent_runtime = self._get_agent_runtime()
        if agent_runtime is None:
            return ApprovalsOrchestratorResult(
                status_code=503,
                payload={"detail": "agent runtime unavailable"},
            )

        async def _approved_executor(
            action_name: str,
            action_args: dict[str, object],
            cid: str,
        ) -> object:
            safe_args = action_args if isinstance(action_args, dict) else {}
            if str(action_name or "").strip() in {"tulpa_run_terminal", "routine_create"}:
                safe_args = {**safe_args, "preapproved": True}
            if hasattr(agent_runtime, "execute_tool"):
                return await agent_runtime.execute_tool(
                    action_name=action_name,
                    action_args=safe_args,
                    customer_id=cid,
                    inject_customer_id=True,
                )
            tools = getattr(agent_runtime, "_tools", {})
            if not isinstance(tools, dict):
                raise RuntimeError("agent runtime does not expose executable tools")
            tool = tools.get(str(action_name or "").strip())
            if tool is None or not hasattr(tool, "ainvoke"):
                raise RuntimeError(f"unknown tool: {action_name}")
            return await tool.ainvoke(safe_args)

        try:
            result = await broker.execute_approved_action(
                approval_id=safe_approval_id,
                customer_id=safe_customer_id,
                executor=_approved_executor,
            )
        except Exception as exc:
            return ApprovalsOrchestratorResult(
                status_code=500,
                payload={"detail": f"execute failed: {exc}"},
            )
        if not bool(result.get("ok")):
            error = str(result.get("error", "")).strip()
            status_code = 400
            if error == "approval_not_found":
                status_code = 404
            elif error == "customer_mismatch":
                status_code = 403
            elif error.startswith("approval_not_executable:"):
                status_code = 409
            logger.warning(
                "Approval execute rejected (status=%s approval_id=%s customer_id=%s error=%s)",
                status_code,
                safe_approval_id,
                safe_customer_id,
                error or "unknown_error",
            )
            return ApprovalsOrchestratorResult(status_code=status_code, payload=result)
        return ApprovalsOrchestratorResult(status_code=200, payload=result)

    def get_approval(self, *, approval_id: str) -> ApprovalsOrchestratorResult:
        payload = self._get_approvals().get(approval_id)
        if payload is None:
            return ApprovalsOrchestratorResult(
                status_code=404,
                payload={"detail": "approval not found"},
            )
        return ApprovalsOrchestratorResult(status_code=200, payload={"ok": True, "approval": payload})

    def pending_status(self, *, customer_id: str, thread_id: str) -> ApprovalsOrchestratorResult:
        cid = str(customer_id or "").strip()
        tid = str(thread_id or "").strip()
        if not cid or not tid:
            return ApprovalsOrchestratorResult(
                status_code=400,
                payload={"detail": "customer_id and thread_id are required"},
            )
        pending = self._get_approvals().has_pending_for_customer_thread(
            customer_id=cid,
            thread_id=tid,
        )
        return ApprovalsOrchestratorResult(status_code=200, payload={"ok": True, "pending": bool(pending)})
