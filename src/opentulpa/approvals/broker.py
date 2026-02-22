"""Approval broker for external-impact side effects."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from typing import Any

from opentulpa.approvals.adapters.base import ApprovalAdapter
from opentulpa.approvals.evaluator import ApprovalEvaluator
from opentulpa.approvals.store import PendingApprovalStore
from opentulpa.core.ids import new_short_id


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


class ApprovalBroker:
    """Orchestrates approvals, persistence, and execution for side-effecting actions.

    Decision matrix (high-level):
    - Gate decisions (`allow`/`require_approval`/`deny`) come from LLM guardrail evaluator output.
    - Instant external actions that evaluate to `require_approval` create a challenge.
    - Scheduled external automations evaluate at routine creation time.
    - Background (`wake_*`) runs are treated as pre-authorized scheduled execution.
      No per-run approval prompts/checks.
    """

    def __init__(
        self,
        *,
        store: PendingApprovalStore,
        runtime: Any | None = None,
        approval_ttl_minutes: int = 10,
        adapters: dict[str, ApprovalAdapter] | None = None,
        text_token_adapter: ApprovalAdapter | None = None,
        origin_resolver: Callable[[str, str], dict[str, Any]] | None = None,
    ) -> None:
        self._store = store
        self._evaluator = ApprovalEvaluator(runtime=runtime)
        self._ttl_seconds = max(60, min(int(approval_ttl_minutes), 120)) * 60
        self._adapters = adapters or {}
        self._text_token_adapter = text_token_adapter
        self._origin_resolver = origin_resolver

    def _resolve_origin(
        self,
        *,
        customer_id: str,
        thread_id: str,
        origin_interface: str | None,
        origin_user_id: str | None,
        origin_conversation_id: str | None,
    ) -> tuple[str, str, str]:
        interface = str(origin_interface or "").strip()
        user_id = str(origin_user_id or "").strip()
        conversation_id = str(origin_conversation_id or "").strip()
        if interface and user_id and conversation_id:
            return interface, user_id, conversation_id
        if self._origin_resolver is not None:
            resolved = self._origin_resolver(customer_id, thread_id)
            if isinstance(resolved, dict):
                interface = _first_non_empty(interface, resolved.get("origin_interface"))
                user_id = _first_non_empty(user_id, resolved.get("origin_user_id"))
                conversation_id = _first_non_empty(
                    conversation_id, resolved.get("origin_conversation_id")
                )
        interface = interface or "unknown"
        user_id = user_id or ""
        conversation_id = conversation_id or ""
        return interface, user_id, conversation_id

    async def _deliver_challenge(self, record: Any) -> str | None:
        adapter = self._adapters.get(record.origin_interface)
        if adapter is not None:
            try:
                sent = await adapter.send_challenge(record)
            except Exception:
                sent = False
            if sent:
                return adapter.name

        if self._text_token_adapter is not None:
            try:
                sent = await self._text_token_adapter.send_challenge(record)
            except Exception:
                sent = False
            if sent:
                return self._text_token_adapter.name
        return None

    async def evaluate_action(
        self,
        *,
        customer_id: str,
        thread_id: str,
        action_name: str,
        action_args: dict[str, Any],
        action_note: str | None = None,
        origin_interface: str | None = None,
        origin_user_id: str | None = None,
        origin_conversation_id: str | None = None,
    ) -> dict[str, Any]:
        interface, user_id, conversation_id = self._resolve_origin(
            customer_id=customer_id,
            thread_id=thread_id,
            origin_interface=origin_interface,
            origin_user_id=origin_user_id,
            origin_conversation_id=origin_conversation_id,
        )
        intent, decision = await self._evaluator.evaluate(
            customer_id=customer_id,
            thread_id=thread_id,
            action_name=action_name,
            action_args=action_args,
            origin_interface=interface,
            origin_user_id=user_id,
            origin_conversation_id=conversation_id,
            action_note=action_note,
        )
        if decision.gate != "require_approval":
            return self._evaluator.as_dict(decision)

        args_json = json.dumps(intent.action_args, sort_keys=True)
        is_background_thread = self._evaluator.is_background_thread(intent.thread_id)
        if is_background_thread and intent.action_name != "routine_create":
            decision.gate = "allow"
            decision.reason = "background_preauthorized_execution"
            decision.status = "approved"
            decision.approval_id = None
            decision.delivery_mode = None
            return self._evaluator.as_dict(decision)

        duplicate = self._store.find_pending_duplicate(
            customer_id=intent.customer_id,
            thread_id=intent.thread_id,
            action_name=intent.action_name,
            action_args_json=args_json,
        )
        if duplicate is not None:
            decision.approval_id = duplicate.id
            decision.status = duplicate.status
            decision.expires_at = duplicate.expires_at
            # Reuse existing pending approval without sending duplicate prompts.
            decision.delivery_mode = "existing_pending"
            return self._evaluator.as_dict(decision)

        # Reuse recent non-pending decisions for exact same action intent.
        recent_same_args = self._store.find_recent_matching(
            customer_id=intent.customer_id,
            thread_id=intent.thread_id,
            action_name=intent.action_name,
            action_args_json=args_json,
            statuses=("approved", "executed"),
            lookback_seconds=self._ttl_seconds,
        )
        if recent_same_args is not None:
            if recent_same_args.status == "approved":
                decision.gate = "allow"
                decision.reason = "reuse_recent_approved"
                decision.status = recent_same_args.status
                decision.expires_at = recent_same_args.expires_at
                return self._evaluator.as_dict(decision)
            # Prevent prompt loops requesting approval repeatedly for the same external action.
            decision.gate = "deny"
            decision.reason = "already_executed_recent_duplicate"
            decision.status = recent_same_args.status
            decision.expires_at = recent_same_args.expires_at
            decision.approval_id = None
            decision.delivery_mode = None
            return self._evaluator.as_dict(decision)

        # Browser tasks often carry transient args (session ids/timeouts). Deduplicate by summary.
        if intent.action_name == "browser_use_run":
            recent_browser = self._store.find_recent_matching(
                customer_id=intent.customer_id,
                thread_id=intent.thread_id,
                action_name=intent.action_name,
                summary=intent.summary,
                statuses=("pending", "approved", "executed"),
                lookback_seconds=self._ttl_seconds,
            )
            if recent_browser is not None:
                if recent_browser.status == "pending":
                    decision.approval_id = recent_browser.id
                    decision.status = recent_browser.status
                    decision.expires_at = recent_browser.expires_at
                    # Reuse existing pending approval without sending duplicate prompts.
                    decision.delivery_mode = "existing_pending"
                    return self._evaluator.as_dict(decision)
                if recent_browser.status == "approved":
                    decision.gate = "allow"
                    decision.reason = "reuse_recent_approved_browser_task"
                    decision.status = recent_browser.status
                    decision.expires_at = recent_browser.expires_at
                    return self._evaluator.as_dict(decision)
                decision.gate = "deny"
                decision.reason = "already_executed_recent_browser_task"
                decision.status = recent_browser.status
                decision.expires_at = recent_browser.expires_at
                decision.approval_id = None
                decision.delivery_mode = None
                return self._evaluator.as_dict(decision)

        approval_id = new_short_id("apr")
        record = self._store.create_pending(
            approval_id=approval_id,
            customer_id=intent.customer_id,
            thread_id=intent.thread_id,
            origin_interface=intent.origin_interface,
            origin_user_id=intent.origin_user_id,
            origin_conversation_id=intent.origin_conversation_id,
            action_name=intent.action_name,
            action_args=intent.action_args,
            recipient_scope=intent.recipient_scope,
            impact_type=intent.impact_type,
            summary=intent.summary,
            reason=intent.reason,
            confidence=float(intent.confidence),
            ttl_seconds=self._ttl_seconds,
        )
        delivery_mode = await self._deliver_challenge(record)
        decision.approval_id = record.id
        decision.status = record.status
        decision.expires_at = record.expires_at
        decision.delivery_mode = delivery_mode
        return self._evaluator.as_dict(decision)

    def get(self, approval_id: str) -> dict[str, Any] | None:
        return self._store.as_dict(self._store.get(approval_id))

    def get_approval_group_status(
        self,
        *,
        approval_id: str,
        window_seconds: int = 60,
    ) -> dict[str, Any] | None:
        record = self._store.get(approval_id)
        if record is None:
            return None
        related = self._store.list_thread_window(
            customer_id=record.customer_id,
            thread_id=record.thread_id,
            anchor_created_at=record.created_at,
            window_seconds=window_seconds,
        )
        if not related:
            related = [record]

        def _parse_dt(value: str) -> datetime:
            dt = datetime.fromisoformat(str(value or "").strip())
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt

        created_points: list[datetime] = []
        for item in related:
            with suppress(Exception):
                created_points.append(_parse_dt(item.created_at))
        group_start = min(created_points) if created_points else _parse_dt(record.created_at)
        deadline = group_start + timedelta(seconds=max(1, int(window_seconds)))
        now = self._store.utc_now()
        window_open = now <= deadline

        ids = [item.id for item in related]
        pending_ids = [item.id for item in related if item.status == "pending"]
        approved_ids = [item.id for item in related if item.status == "approved"]
        executed_ids = [item.id for item in related if item.status == "executed"]
        denied_ids = [item.id for item in related if item.status == "denied"]
        expired_ids = [item.id for item in related if item.status == "expired"]

        executable_ids: list[str] = []
        if window_open and not pending_ids and not denied_ids and not expired_ids:
            executable_ids = approved_ids

        return {
            "window_seconds": int(window_seconds),
            "window_open": bool(window_open),
            "group_start_at": group_start.isoformat(),
            "deadline_at": deadline.isoformat(),
            "all_ids": ids,
            "pending_ids": pending_ids,
            "approved_ids": approved_ids,
            "executed_ids": executed_ids,
            "denied_ids": denied_ids,
            "expired_ids": expired_ids,
            "executable_ids": executable_ids,
        }

    async def decide(
        self,
        *,
        approval_id: str,
        decision: str,
        actor_interface: str,
        actor_id: str,
    ) -> dict[str, Any]:
        record = self._store.get(approval_id)
        if record is None:
            return {"ok": False, "status": "denied", "reason": "approval_not_found"}
        if record.status != "pending":
            return {
                "ok": False,
                "status": "denied",
                "reason": f"approval_not_pending:{record.status}",
                "approval_id": record.id,
            }
        if str(actor_interface or "").strip() != str(record.origin_interface or "").strip():
            return {
                "ok": False,
                "status": "denied",
                "reason": "wrong_interface",
                "approval_id": record.id,
            }
        expected_actor = str(record.origin_user_id or "").strip()
        provided_actor = str(actor_id or "").strip()
        if not expected_actor or provided_actor != expected_actor:
            return {
                "ok": False,
                "status": "denied",
                "reason": "unauthorized_actor",
                "approval_id": record.id,
            }
        choice = str(decision or "").strip().lower()
        if choice not in {"approve", "deny"}:
            return {
                "ok": False,
                "status": "denied",
                "reason": "invalid_decision",
                "approval_id": record.id,
            }
        updated = self._store.set_decision(
            approval_id=record.id,
            decision=choice,
            actor_id=provided_actor,
        )
        payload = self._store.as_dict(updated)
        if payload is None:
            return {"ok": False, "status": "denied", "reason": "decision_persist_failed"}
        payload["ok"] = True
        payload["decision"] = choice
        return payload

    async def execute_approved_action(
        self,
        *,
        approval_id: str,
        customer_id: str,
        executor: Callable[[str, dict[str, Any], str], Awaitable[Any]],
    ) -> dict[str, Any]:
        record = self._store.get(approval_id)
        if record is None:
            return {"ok": False, "error": "approval_not_found"}
        if str(record.customer_id) != str(customer_id):
            return {"ok": False, "error": "customer_mismatch"}
        if record.status == "executed":
            # Idempotent replay handling: a duplicate execute call should be safe.
            return {
                "ok": True,
                "approval_id": record.id,
                "status": "executed",
                "action_name": record.action_name,
                "already_executed": True,
            }
        if record.status != "approved":
            return {"ok": False, "error": f"approval_not_executable:{record.status}"}

        try:
            action_args = json.loads(record.action_args_json)
        except Exception:
            action_args = {}
        if not isinstance(action_args, dict):
            action_args = {}

        result = await executor(record.action_name, action_args, record.customer_id)
        updated = self._store.mark_executed(record.id)
        return {
            "ok": bool(updated and updated.status == "executed"),
            "approval_id": record.id,
            "status": updated.status if updated else "approved",
            "action_name": record.action_name,
            "result": result,
        }
