"""Approval broker for external-impact side effects."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any

from opentulpa.approvals.adapters.base import ApprovalAdapter
from opentulpa.approvals.models import ActionIntent, GateDecision, RecipientScope
from opentulpa.approvals.policy import INTERNAL_ONLY_ACTIONS, evaluate_policy
from opentulpa.approvals.store import PendingApprovalStore
from opentulpa.core.ids import new_short_id

READ_ACTIONS: set[str] = {
    "web_search",
    "fetch_link_content",
    "uploaded_file_search",
    "uploaded_file_get",
    "uploaded_file_analyze",
    "browser_use_task_get",
    "routine_list",
    "task_status",
    "task_events",
    "task_artifacts",
    "tulpa_read_file",
    "tulpa_catalog",
    "server_time",
}

WRITE_ACTIONS: set[str] = {
    "uploaded_file_send",
    "web_image_send",
    "browser_use_run",
    "browser_use_task_control",
    "tulpa_write_file",
    "tulpa_run_terminal",
    "routine_create",
    "routine_delete",
    "automation_delete",
    "task_relaunch",
    "task_cancel",
}

EXTERNAL_DEFAULT_ACTIONS: set[str] = {
    "slack_post",
    "whatsapp_send",
    "email_send",
}

_SENSITIVE_KEY_PARTS = {"key", "token", "secret", "password", "authorization", "api"}
_TERMINAL_POST_HINTS = (
    "post",
    "sendmessage",
    "chat.postmessage",
    "smtp",
    "mailgun",
    "twilio",
    "whatsapp",
    "discord",
    "telegram",
    "publish",
)
_TERMINAL_NETWORK_HINTS = (
    "curl ",
    "wget ",
    "http://",
    "https://",
    "requests",
    "httpx",
    "urllib",
    "aiohttp",
    "smtp",
    "imap",
    "pop3",
    "slack",
    "whatsapp",
    "discord",
    "telegram",
    "mailgun",
    "twilio",
)
_PURCHASE_HINTS = ("buy", "checkout", "purchase", "invoice", "payment", "charge", "transfer")
_BROWSER_WRITE_HINTS = (
    "login",
    "log in",
    "sign in",
    "submit",
    "post",
    "publish",
    "send",
    "message",
    "comment",
    "upload",
    "update",
    "edit",
    "delete",
    "create",
    "book",
    "reserve",
)


def _parse_impact(value: str, default: str = "write") -> str:
    raw = str(value or "").strip().lower()
    if raw in {"read", "write", "purchase", "costly"}:
        return raw
    return default


def _parse_scope(value: str, default: RecipientScope = "unknown") -> RecipientScope:
    raw = str(value or "").strip().lower()
    if raw in {"self", "external", "unknown"}:
        return raw  # type: ignore[return-value]
    return default


def _mask_sensitive(action_args: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in (action_args or {}).items():
        key_text = str(key).strip()
        key_lower = key_text.lower()
        if any(part in key_lower for part in _SENSITIVE_KEY_PARTS):
            out[key_text] = "***"
            continue
        if isinstance(value, str):
            out[key_text] = value[:400]
        elif isinstance(value, (int, float, bool)) or value is None:
            out[key_text] = value
        elif isinstance(value, list):
            out[key_text] = [str(item)[:120] for item in value[:10]]
        elif isinstance(value, dict):
            out[key_text] = {str(k)[:40]: str(v)[:120] for k, v in list(value.items())[:10]}
        else:
            out[key_text] = str(value)[:200]
    return out


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


class ApprovalBroker:
    """Evaluates tool-call intent and mediates approval challenges."""

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
        self._runtime = runtime
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

    @staticmethod
    def _resolve_recipient_scope(
        *,
        action_name: str,
        action_args: dict[str, Any],
        origin_conversation_id: str,
        origin_user_id: str,
    ) -> RecipientScope:
        if action_name in {"uploaded_file_send", "web_image_send", "tulpa_write_file"}:
            return "self"
        if action_name in EXTERNAL_DEFAULT_ACTIONS:
            return "external"

        destination = ""
        for key in (
            "chat_id",
            "channel_id",
            "destination",
            "recipient",
            "to",
            "email",
            "phone",
            "conversation_id",
            "target_user_id",
            "user_id",
        ):
            value = action_args.get(key)
            if value is None:
                continue
            destination = str(value).strip()
            if destination:
                break
        if not destination:
            return "unknown"
        if destination in {str(origin_conversation_id).strip(), str(origin_user_id).strip()}:
            return "self"
        return "external"

    @staticmethod
    def _classify_terminal_command(command: str) -> tuple[str, RecipientScope]:
        cmd = str(command or "").strip().lower()
        if not cmd:
            return "write", "unknown"
        if any(hint in cmd for hint in _PURCHASE_HINTS):
            return "purchase", "unknown"
        if "curl " in cmd or "wget " in cmd or "http://" in cmd or "https://" in cmd:
            if any(hint in cmd for hint in _TERMINAL_POST_HINTS):
                return "write", "unknown"
            if re.search(r"\s-x\s+post\b", cmd) or "--data" in cmd:
                return "write", "unknown"
            return "read", "unknown"
        return "write", "self"

    @staticmethod
    def _terminal_has_external_hint(command: str) -> bool:
        cmd = str(command or "").strip().lower()
        if not cmd:
            return False
        return any(hint in cmd for hint in _TERMINAL_NETWORK_HINTS)

    @staticmethod
    def _classify_browser_task(task: str) -> str:
        text = str(task or "").strip().lower()
        if not text:
            return "write"
        if any(hint in text for hint in _PURCHASE_HINTS):
            return "purchase"
        if any(hint in text for hint in _BROWSER_WRITE_HINTS):
            return "write"
        return "read"

    async def _llm_intent_hint(
        self,
        *,
        action_name: str,
        action_args: dict[str, Any],
    ) -> dict[str, Any]:
        if self._runtime is None or not hasattr(self._runtime, "classify_guardrail_intent"):
            return {"ok": False, "error": "runtime_unavailable"}
        try:
            result = await self._runtime.classify_guardrail_intent(
                action_name=action_name,
                action_args=_mask_sensitive(action_args),
            )
        except Exception as exc:
            return {"ok": False, "error": f"classifier_error:{exc}"}
        if not isinstance(result, dict):
            return {"ok": False, "error": "classifier_invalid_payload"}
        if not bool(result.get("ok", True)):
            return {"ok": False, "error": str(result.get("error", "classifier_not_ok"))}
        return {"ok": True, **result}

    @staticmethod
    def _summarize_action(action_name: str, action_args: dict[str, Any]) -> str:
        if action_name == "uploaded_file_send":
            return f"send file_id={str(action_args.get('file_id', '')).strip()[:60]}"
        if action_name == "web_image_send":
            return f"send image from url={str(action_args.get('url', '')).strip()[:100]}"
        if action_name == "browser_use_run":
            return f"run browser task: {str(action_args.get('task', '')).strip()[:160]}"
        if action_name == "browser_use_task_control":
            return (
                f"browser task control task_id={str(action_args.get('task_id', '')).strip()[:50]} "
                f"action={str(action_args.get('action', 'stop_task_and_session')).strip()[:30]}"
            )
        if action_name == "tulpa_run_terminal":
            return f"run terminal command: {str(action_args.get('command', '')).strip()[:180]}"
        return f"execute {action_name}"

    async def _build_intent(
        self,
        *,
        customer_id: str,
        thread_id: str,
        action_name: str,
        action_args: dict[str, Any],
        origin_interface: str,
        origin_user_id: str,
        origin_conversation_id: str,
    ) -> ActionIntent:
        recipient_scope = self._resolve_recipient_scope(
            action_name=action_name,
            action_args=action_args,
            origin_conversation_id=origin_conversation_id,
            origin_user_id=origin_user_id,
        )

        if action_name in INTERNAL_ONLY_ACTIONS or action_name in READ_ACTIONS:
            impact = "read"
        elif action_name in WRITE_ACTIONS:
            impact = "write"
        else:
            impact = "write"

        if action_name == "browser_use_run":
            impact = self._classify_browser_task(str(action_args.get("task", "")))
            recipient_scope = "unknown"
        elif action_name == "browser_use_task_control":
            control_action = str(action_args.get("action", "stop_task_and_session")).strip().lower()
            if control_action in {"stop", "pause", "stop_task_and_session"}:
                impact = "read"
            else:
                impact = "write"
            recipient_scope = "unknown"
        elif action_name == "tulpa_run_terminal":
            impact, recipient_scope = self._classify_terminal_command(
                str(action_args.get("command", ""))
            )

        reason = "policy_matrix"
        confidence = 0.7
        llm_uncertain = False
        should_llm_classify = action_name in {"browser_use_run", "browser_use_task_control"}
        if action_name == "tulpa_run_terminal":
            cmd = str(action_args.get("command", ""))
            should_llm_classify = self._terminal_has_external_hint(cmd)
            if not should_llm_classify:
                # Local terminal commands are internal/self-impact and should not
                # trigger external-impact approvals.
                reason = "local_terminal_command"
                confidence = max(confidence, 0.9)
        if should_llm_classify:
            hint = await self._llm_intent_hint(action_name=action_name, action_args=action_args)
            if hint.get("ok"):
                impact = _parse_impact(hint.get("impact_type"), default=impact)
                recipient_scope = _parse_scope(hint.get("recipient_scope"), default=recipient_scope)
                reason = _first_non_empty(hint.get("reason"), reason)[:500]
                with_conf = hint.get("confidence", confidence)
                try:
                    confidence = float(with_conf)
                except Exception:
                    confidence = 0.6
                confidence = max(0.0, min(confidence, 1.0))
            else:
                reason = str(hint.get("error", "guardrail_classifier_failed"))[:500]
                llm_uncertain = True

        summary = self._summarize_action(action_name, action_args)
        return ActionIntent(
            customer_id=str(customer_id or "").strip(),
            thread_id=str(thread_id or "").strip(),
            action_name=str(action_name or "").strip(),
            action_args=action_args if isinstance(action_args, dict) else {},
            origin_interface=origin_interface,
            origin_user_id=origin_user_id,
            origin_conversation_id=origin_conversation_id,
            recipient_scope=recipient_scope,
            impact_type=_parse_impact(impact),  # type: ignore[arg-type]
            summary=summary,
            reason=reason,
            confidence=confidence,
            llm_uncertain=llm_uncertain,
        )

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
        intent = await self._build_intent(
            customer_id=customer_id,
            thread_id=thread_id,
            action_name=action_name,
            action_args=action_args,
            origin_interface=interface,
            origin_user_id=user_id,
            origin_conversation_id=conversation_id,
        )
        policy = evaluate_policy(intent)
        decision = GateDecision(
            gate=policy.gate,
            reason=policy.reason,
            summary=intent.summary,
            confidence=intent.confidence,
            recipient_scope=intent.recipient_scope,
            impact_type=intent.impact_type,
        )
        if decision.gate != "require_approval":
            return asdict(decision)

        args_json = json.dumps(intent.action_args, sort_keys=True)
        duplicate = self._store.find_pending_duplicate(
            customer_id=intent.customer_id,
            thread_id=intent.thread_id,
            action_name=intent.action_name,
            action_args_json=args_json,
        )
        if duplicate is not None:
            delivery_mode = await self._deliver_challenge(duplicate)
            decision.approval_id = duplicate.id
            decision.status = duplicate.status
            decision.expires_at = duplicate.expires_at
            decision.delivery_mode = delivery_mode
            return asdict(decision)

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
                return asdict(decision)
            # Prevent prompt loops requesting approval repeatedly for the same external action.
            decision.gate = "deny"
            decision.reason = "already_executed_recent_duplicate"
            decision.status = recent_same_args.status
            decision.expires_at = recent_same_args.expires_at
            decision.approval_id = None
            decision.delivery_mode = None
            return asdict(decision)

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
                    delivery_mode = await self._deliver_challenge(recent_browser)
                    decision.approval_id = recent_browser.id
                    decision.status = recent_browser.status
                    decision.expires_at = recent_browser.expires_at
                    decision.delivery_mode = delivery_mode
                    return asdict(decision)
                if recent_browser.status == "approved":
                    decision.gate = "allow"
                    decision.reason = "reuse_recent_approved_browser_task"
                    decision.status = recent_browser.status
                    decision.expires_at = recent_browser.expires_at
                    return asdict(decision)
                decision.gate = "deny"
                decision.reason = "already_executed_recent_browser_task"
                decision.status = recent_browser.status
                decision.expires_at = recent_browser.expires_at
                decision.approval_id = None
                decision.delivery_mode = None
                return asdict(decision)

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
        return asdict(decision)

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
            return {"ok": False, "status": "denied", "reason": "wrong_interface", "approval_id": record.id}
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
            return {"ok": False, "status": "denied", "reason": "invalid_decision", "approval_id": record.id}
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
