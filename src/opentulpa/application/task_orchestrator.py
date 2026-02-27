"""Application-layer orchestration for task APIs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opentulpa.application.contracts import ApplicationResult


class TaskOrchestratorResult(ApplicationResult[dict[str, object]]):
    """Normalized route-friendly result payload."""


class TaskOrchestrator:
    """Owns task API business validation and service interaction."""

    def __init__(self, *, get_tasks: Callable[[], Any]) -> None:
        self._get_tasks = get_tasks

    async def create_task(
        self,
        *,
        customer_id: str,
        goal: str,
        payload: dict[str, object],
        risk_level: str,
        idempotency_key: str | None,
    ) -> TaskOrchestratorResult:
        cid = str(customer_id or "").strip()
        objective = str(goal or "").strip()
        safe_payload = payload if isinstance(payload, dict) else {}
        safe_risk_level = str(risk_level or "").strip() or "low"

        if not cid or not objective:
            return TaskOrchestratorResult(
                status_code=400,
                payload={"detail": "customer_id and goal are required"},
            )
        if "steps" in safe_payload:
            steps = safe_payload.get("steps")
            if not isinstance(steps, list):
                return TaskOrchestratorResult(
                    status_code=400,
                    payload={"detail": "payload.steps must be a list"},
                )
            bad_idx = next((i for i, step in enumerate(steps) if not isinstance(step, dict)), None)
            if bad_idx is not None:
                return TaskOrchestratorResult(
                    status_code=400,
                    payload={
                        "detail": (
                            "payload.steps entries must be objects with a 'type' field "
                            "(e.g. {'type':'run_terminal', ...})."
                        ),
                        "bad_step_index": bad_idx,
                    },
                )

        task = await self._get_tasks().create_task(
            customer_id=cid,
            goal=objective,
            payload=safe_payload,
            risk_level=safe_risk_level,
            idempotency_key=idempotency_key,
        )
        return TaskOrchestratorResult(status_code=200, payload={"ok": True, "task": task})

    def get_task(self, *, task_id: str) -> TaskOrchestratorResult:
        try:
            task = self._get_tasks().get_task(task_id)
        except KeyError:
            return TaskOrchestratorResult(status_code=404, payload={"detail": "task not found"})
        return TaskOrchestratorResult(status_code=200, payload={"ok": True, "task": task})

    def list_events(self, *, task_id: str, limit: int, offset: int) -> TaskOrchestratorResult:
        try:
            self._get_tasks().get_task(task_id)
        except KeyError:
            return TaskOrchestratorResult(status_code=404, payload={"detail": "task not found"})
        events = self._get_tasks().list_events(task_id, limit=limit, offset=offset)
        return TaskOrchestratorResult(status_code=200, payload={"ok": True, "events": events})

    def list_artifacts(self, *, task_id: str) -> TaskOrchestratorResult:
        try:
            self._get_tasks().get_task(task_id)
        except KeyError:
            return TaskOrchestratorResult(status_code=404, payload={"detail": "task not found"})
        artifacts = self._get_tasks().list_task_artifacts(task_id)
        return TaskOrchestratorResult(status_code=200, payload={"ok": True, "artifacts": artifacts})

    async def relaunch_task(
        self,
        *,
        task_id: str,
        clarification: Any | None,
        trigger_reason: str,
    ) -> TaskOrchestratorResult:
        safe_reason = str(trigger_reason or "").strip() or "user_requested"
        try:
            task = await self._get_tasks().relaunch_task(
                task_id=task_id,
                trigger_reason=safe_reason,
                clarification=clarification,
            )
        except KeyError:
            return TaskOrchestratorResult(status_code=404, payload={"detail": "task not found"})
        return TaskOrchestratorResult(status_code=200, payload={"ok": True, "task": task})

    async def cancel_task(self, *, task_id: str) -> TaskOrchestratorResult:
        try:
            task = await self._get_tasks().cancel_task(task_id)
        except KeyError:
            return TaskOrchestratorResult(status_code=404, payload={"detail": "task not found"})
        return TaskOrchestratorResult(status_code=200, payload={"ok": True, "task": task})
