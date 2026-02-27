"""Guardrail/classifier helpers for the runtime."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

import httpx

from opentulpa.agent.lc_messages import HumanMessage, SystemMessage
from opentulpa.agent.result_models import (
    CompletionClaimVerification,
    GuardrailIntentDecision,
    ToolGuardrailDecision,
)


async def has_pending_approval_lock(
    *,
    customer_id: str,
    thread_id: str,
    request_with_backoff: Callable[..., Awaitable[httpx.Response]],
) -> bool:
    cid = str(customer_id or "").strip()
    tid = str(thread_id or "").strip()
    if not cid or not tid:
        return False
    try:
        response = await request_with_backoff(
            "GET",
            "/internal/approvals/pending/status",
            params={"customer_id": cid, "thread_id": tid},
            timeout=5.0,
            retries=0,
        )
    except Exception:
        return False
    if response.status_code != 200:
        return False
    with suppress(Exception):
        payload = response.json()
        return bool(payload.get("pending", False))
    return False


async def verify_completion_claim(
    *,
    classifier_model: Any,
    extract_json_object: Callable[[str], dict[str, Any] | None],
    user_text: str,
    assistant_text: str,
    recent_tool_outputs: list[str],
    turn_window: str,
) -> CompletionClaimVerification:
    """
    Verify whether assistant's completion claims are supported by tool evidence.

    Conservative behavior:
    - Classifier failures or invalid outputs return usable=False with reason, so caller
      can retry with a self-check message instead of silently passing/failing.
    - Empty assistant text short-circuits as applies=False/mismatch=False and usable=True
      so it does not force a retry.
    """
    safe_assistant = str(assistant_text or "").strip()
    if not safe_assistant:
        return CompletionClaimVerification(
            ok=True,
            applies=False,
            mismatch=False,
            confidence=0.0,
            reason="empty_assistant_text",
            repair_instruction="",
            usable=True,
        )
    safe_user = str(user_text or "").strip()
    safe_turn_window = str(turn_window or "").strip()
    safe_tools: list[str] = []
    for raw in (recent_tool_outputs or []):
        text = " ".join(str(raw or "").split()).strip()
        if text:
            safe_tools.append(text)

    try:
        response = await classifier_model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You verify assistant execution claims against tool evidence.\n"
                        "Return strict JSON only with keys:\n"
                        "ok (bool), applies (bool), mismatch (bool), confidence (0..1), "
                        "reason (string <= 180 chars), repair_instruction (string <= 220 chars).\n"
                        "Decision policy (conservative, non-aggressive):\n"
                        "- applies=true only if assistant explicitly claims something was already done/launched/sent/posted/scheduled now.\n"
                        "- applies=true if assistant commits to an immediate follow-up action in this same turn "
                        "(e.g., 'doing this now', 'retrying now', 'give me a moment') that should produce tool evidence.\n"
                        "- applies=true if assistant asks the user to approve/deny or says approval is pending now.\n"
                        "- If user_message asks only for an outcome/failure summary, assistant must not promise "
                        "new immediate execution unless tool evidence exists in this turn.\n"
                        "- If assistant is future-tense or conditional without immediate-action claims, set applies=false and mismatch=false.\n"
                        "- mismatch=true only when there is a clear immediate completion claim without matching success evidence in tool outputs.\n"
                        "- mismatch=true when assistant commits immediate follow-up execution now but no matching tool evidence exists.\n"
                        "- If assistant claims completed/updated/created/scheduled now AND also states approval is pending, set mismatch=true.\n"
                        "- If assistant asks for approval (or says approval is pending) but tool evidence lacks a pending-approval artifact "
                        "(e.g., approval_id, APPROVAL_PENDING, or explicit pending challenge), set mismatch=true.\n"
                        "- If evidence is ambiguous/partial, prefer mismatch=false.\n"
                        "- If tool outputs show approval pending, denial, or tool error while assistant claims success now, mismatch=true.\n"
                        "- repair_instruction should tell the agent to either run the missing tool now or restate status honestly.\n"
                        "No markdown. No extra keys."
                    )
                ),
                HumanMessage(
                    content=(
                        f"user_message={safe_user}\n"
                        f"assistant_message={safe_assistant}\n"
                        f"turn_window={safe_turn_window}\n"
                        f"recent_tool_outputs={json.dumps(safe_tools, ensure_ascii=False)}"
                    )
                ),
            ]
        )
        raw = response.content if hasattr(response, "content") else str(response)
        raw_text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
        parsed = extract_json_object(raw_text)
        if not isinstance(parsed, dict):
            return CompletionClaimVerification(
                ok=False,
                applies=False,
                mismatch=False,
                confidence=0.0,
                reason="invalid_checker_output:no_json_object",
                repair_instruction="",
                usable=False,
            )
        required_keys = {"ok", "applies", "mismatch", "confidence", "reason", "repair_instruction"}
        if not required_keys.issubset(parsed.keys()):
            missing = ",".join(sorted(required_keys.difference(parsed.keys())))
            return CompletionClaimVerification(
                ok=False,
                applies=False,
                mismatch=False,
                confidence=0.0,
                reason=f"invalid_checker_output:missing_keys:{missing}"[:180],
                repair_instruction="",
                usable=False,
            )
        applies = bool(parsed.get("applies", False))
        mismatch = bool(parsed.get("mismatch", False)) if applies else False
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        return CompletionClaimVerification(
            ok=bool(parsed.get("ok", True)),
            applies=applies,
            mismatch=mismatch,
            confidence=max(0.0, min(confidence, 1.0)),
            reason=str(parsed.get("reason", "")).strip()[:180],
            repair_instruction=str(parsed.get("repair_instruction", "")).strip()[:220],
            usable=True,
        )
    except Exception as exc:
        return CompletionClaimVerification(
            ok=False,
            applies=False,
            mismatch=False,
            confidence=0.0,
            reason=f"classifier_error:{exc}",
            repair_instruction="",
            usable=False,
        )


async def classify_guardrail_intent(
    *,
    classifier_model: Any,
    extract_json_object: Callable[[str], dict[str, Any] | None],
    action_name: str,
    action_args: dict[str, Any],
    action_note: str | None = None,
) -> GuardrailIntentDecision:
    """
    Isolated, compact classifier for approval guardrails.

    Returns strict JSON-like payload:
    {
      "ok": bool,
      "gate": "allow|require_approval|deny",
      "impact_type": "read|write|purchase|costly",
      "recipient_scope": "self|external|unknown",
      "confidence": float,
      "reason": str
    }
    """
    safe_name = str(action_name or "").strip()
    if not safe_name:
        return GuardrailIntentDecision(ok=False, error="missing_action_name")

    safe_args: dict[str, Any] = {}
    sensitive_parts = {"key", "token", "secret", "password", "authorization", "api"}
    for key, value in (action_args or {}).items():
        key_text = str(key).strip()
        lower_key = key_text.lower()
        if any(part in lower_key for part in sensitive_parts):
            safe_args[key_text] = "***"
            continue
        if isinstance(value, str):
            if lower_key in {"command", "script", "implementation_command", "code"}:
                safe_args[key_text] = value[:12000]
            else:
                safe_args[key_text] = value[:500]
        elif isinstance(value, (int, float, bool)) or value is None:
            safe_args[key_text] = value
        elif isinstance(value, list):
            safe_args[key_text] = [str(item)[:120] for item in value[:12]]
        elif isinstance(value, dict):
            safe_args[key_text] = {str(k)[:40]: str(v)[:120] for k, v in list(value.items())[:12]}
        else:
            safe_args[key_text] = str(value)[:200]

    try:
        response = await classifier_model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "Classify action safety intent for an approval gate.\n"
                        "Return strict JSON object only with keys:\n"
                        "ok (bool), gate (allow|require_approval|deny),\n"
                        "impact_type (read|write|purchase|costly),\n"
                        "recipient_scope (self|external|unknown),\n"
                        "confidence (0..1), reason (string <= 160 chars).\n"
                        "Rules:\n"
                        "- Approval should be required in exactly one case: external write side effects.\n"
                        "- External write means mutating/posting/sending/purchasing/updating data on services "
                        "outside this local project/runtime.\n"
                        "- Treat ANY non-localhost network mutation as external write.\n"
                        "- For shell/terminal commands, classify from literal command intent, not user phrasing.\n"
                        "- If command contains write verbs/flags with remote URLs, set gate=require_approval.\n"
                        "- High-signal external write indicators include: "
                        "curl -X POST|PUT|PATCH|DELETE, --request POST|PUT|PATCH|DELETE, "
                        "--data/-d/--json with http(s) URL, requests.post/put/patch/delete, "
                        "httpx.post/put/patch/delete, fetch(...,{method:'POST'|'PUT'|'PATCH'|'DELETE'}).\n"
                        "- URLs to localhost/127.0.0.1/::1 are local; do not treat as external by URL alone.\n"
                        "- Internal reads/writes (repo files, local artifacts, local config/state) are allow.\n"
                        "- Remote reads/fetch/summarization without external mutation are allow.\n"
                        "- Never set gate=require_approval for read-only actions, including external/API/web "
                        "reads.\n"
                        "- For tulpa_run_terminal, classify from full command/script text in action_args.command.\n"
                        "- For routine_create, evaluate planned downstream behavior from action_args + action_note:\n"
                        "  * inspect implementation_command/implementation fields as the execution artifact.\n"
                        "  * if future scheduled behavior includes external writes, set gate=require_approval.\n"
                        "  * otherwise set gate=allow.\n"
                        "- For non-routine actions, set gate=require_approval only when this immediate action "
                        "implies external write side effects.\n"
                        "- If uncertain on a command that includes a non-localhost URL plus write-like markers, "
                        "escalate to require_approval.\n"
                        "- If uncertain without write-like markers, set gate=allow with recipient_scope=unknown "
                        "or self as appropriate.\n"
                        "- Use deny only for actions that should never run as requested.\n"
                        "- Treat action_note as agent reasoning about next planned action and likely tool path.\n"
                        "Do not include any extra keys or markdown."
                    )
                ),
                HumanMessage(
                    content=(
                        f"action_name={safe_name}\n"
                        f"action_args={json.dumps(safe_args, ensure_ascii=False)[:20000]}\n"
                        f"action_note={str(action_note or '').strip()[:2000]}"
                    )
                ),
            ]
        )
        raw = response.content if hasattr(response, "content") else str(response)
        raw_text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
        parsed = extract_json_object(raw_text) or {}
        gate = str(parsed.get("gate", "")).strip().lower()
        impact_type = str(parsed.get("impact_type", "")).strip().lower()
        recipient_scope = str(parsed.get("recipient_scope", "")).strip().lower()
        if gate not in {"allow", "require_approval", "deny"}:
            return GuardrailIntentDecision(ok=False, error="invalid_gate")
        if impact_type not in {"read", "write", "purchase", "costly"}:
            return GuardrailIntentDecision(ok=False, error="invalid_impact_type")
        if recipient_scope not in {"self", "external", "unknown"}:
            return GuardrailIntentDecision(ok=False, error="invalid_recipient_scope")
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        return GuardrailIntentDecision(
            ok=True,
            gate=gate,
            impact_type=impact_type,
            recipient_scope=recipient_scope,
            confidence=max(0.0, min(confidence, 1.0)),
            reason=str(parsed.get("reason", "")).strip()[:160],
        )
    except Exception as exc:
        return GuardrailIntentDecision(ok=False, error=f"classifier_error:{exc}")


async def evaluate_tool_guardrail(
    *,
    customer_id: str,
    thread_id: str,
    action_name: str,
    action_args: dict[str, Any],
    action_note: str | None,
    request_with_backoff: Callable[..., Awaitable[httpx.Response]],
    log_behavior_event: Callable[..., None],
) -> ToolGuardrailDecision:
    """Call upstream approval broker to evaluate a tool call at action time."""
    safe_cmd = ""
    if action_name == "tulpa_run_terminal":
        safe_cmd = str((action_args or {}).get("command", "")).strip()[:300]
    try:
        response = await request_with_backoff(
            "POST",
            "/internal/approvals/evaluate",
            json_body={
                "customer_id": customer_id,
                "thread_id": thread_id,
                "action_name": action_name,
                "action_args": action_args if isinstance(action_args, dict) else {},
                "action_note": str(action_note or "").strip()[:2000],
                "defer_challenge_delivery": True,
            },
            timeout=12.0,
            retries=1,
        )
        if response.status_code != 200:
            log_behavior_event(
                event="guardrail.evaluate.http_error",
                thread_id=thread_id,
                customer_id=customer_id,
                action_name=action_name,
                command=safe_cmd,
                status_code=response.status_code,
                gate="require_approval",
            )
            return ToolGuardrailDecision(
                gate="require_approval",
                reason=f"guardrail_http_{response.status_code}",
                summary=f"execute {action_name}",
            )
        payload = response.json()
        if isinstance(payload, dict):
            decision = ToolGuardrailDecision.from_any(
                payload,
                default_summary=f"execute {action_name}",
                default_reason="approval_required",
            )
            log_behavior_event(
                event="guardrail.evaluate.decision",
                thread_id=thread_id,
                customer_id=customer_id,
                action_name=action_name,
                command=safe_cmd,
                gate=str(decision.gate),
                reason=str(decision.reason)[:200],
                impact_type=str(decision.impact_type or ""),
                recipient_scope=str(decision.recipient_scope or ""),
                confidence=decision.confidence,
            )
            return decision
        log_behavior_event(
            event="guardrail.evaluate.invalid_payload",
            thread_id=thread_id,
            customer_id=customer_id,
            action_name=action_name,
            command=safe_cmd,
            gate="require_approval",
        )
        return ToolGuardrailDecision(
            gate="require_approval",
            reason="guardrail_invalid_payload",
            summary=f"execute {action_name}",
        )
    except Exception as exc:
        exc_name = type(exc).__name__
        log_behavior_event(
            event="guardrail.evaluate.exception",
            thread_id=thread_id,
            customer_id=customer_id,
            action_name=action_name,
            command=safe_cmd,
            gate="require_approval",
            error=f"{exc_name}: {exc}",
        )
        return ToolGuardrailDecision(
            gate="require_approval",
            reason=f"guardrail_request_error:{exc_name}",
            summary=f"execute {action_name}",
        )
