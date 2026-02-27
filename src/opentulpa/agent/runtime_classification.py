"""Classifier and guardrail runtime facades."""

from __future__ import annotations

from typing import Any

from opentulpa.agent.runtime_guardrails import (
    classify_guardrail_intent as _classify_guardrail_intent,
)
from opentulpa.agent.runtime_guardrails import (
    evaluate_tool_guardrail as _evaluate_tool_guardrail,
)
from opentulpa.agent.runtime_guardrails import (
    verify_completion_claim as _verify_completion_claim,
)
from opentulpa.agent.runtime_wake import (
    classify_wake_event as _classify_wake_event,
)


async def classify_wake_event(
    runtime: Any,
    *,
    customer_id: str,
    event_label: str,
    payload: dict[str, Any],
) -> Any:
    return await _classify_wake_event(
        classifier_model=runtime._wake_classifier_model,
        extract_json_object=runtime._extract_json_object,
        customer_id=customer_id,
        event_label=event_label,
        payload=payload,
    )


async def verify_completion_claim(
    runtime: Any,
    *,
    user_text: str,
    assistant_text: str,
    recent_tool_outputs: list[str],
    turn_window: str | None = None,
) -> Any:
    return await _verify_completion_claim(
        classifier_model=runtime._guardrail_classifier_model,
        extract_json_object=runtime._extract_json_object,
        user_text=user_text,
        assistant_text=assistant_text,
        recent_tool_outputs=recent_tool_outputs,
        turn_window=str(turn_window or ""),
    )


async def classify_guardrail_intent(
    runtime: Any,
    *,
    action_name: str,
    action_args: dict[str, Any],
    action_note: str | None = None,
) -> Any:
    return await _classify_guardrail_intent(
        classifier_model=runtime._guardrail_classifier_model,
        extract_json_object=runtime._extract_json_object,
        action_name=action_name,
        action_args=action_args,
        action_note=action_note,
    )


async def evaluate_tool_guardrail(
    runtime: Any,
    *,
    customer_id: str,
    thread_id: str,
    action_name: str,
    action_args: dict[str, Any],
    action_note: str | None = None,
) -> Any:
    return await _evaluate_tool_guardrail(
        customer_id=customer_id,
        thread_id=thread_id,
        action_name=action_name,
        action_args=action_args,
        action_note=action_note,
        request_with_backoff=runtime._request_with_backoff,
        log_behavior_event=runtime.log_behavior_event,
    )
