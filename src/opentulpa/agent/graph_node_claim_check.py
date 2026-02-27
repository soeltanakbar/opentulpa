"""Claim-check graph node implementation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from opentulpa.agent.claim_check import (
    collect_recent_tool_outputs as _collect_recent_tool_outputs,
)
from opentulpa.agent.claim_check import latest_turn_messages as _latest_turn_messages
from opentulpa.agent.claim_check import retry_backoff_seconds as _retry_backoff_seconds
from opentulpa.agent.claim_check import serialize_turn_window as _serialize_turn_window
from opentulpa.agent.claim_check import trim_text_to_token_budget as _trim_text_to_token_budget
from opentulpa.agent.graph_node_limits import (
    compute_claim_check_retry_limit,
    compute_empty_output_retry_limit,
)
from opentulpa.agent.lc_messages import AIMessage, SystemMessage
from opentulpa.agent.models import AgentState
from opentulpa.agent.result_models import CompletionClaimVerification
from opentulpa.agent.utils import (
    content_to_text as _content_to_text,
)


async def claim_check_node(
    state: AgentState,
    *,
    runtime: Any,
    log: Callable[..., None],
) -> dict[str, Any]:
    messages = state.get("messages", [])
    if not messages:
        return {"claim_check_needs_retry": False}
    last = messages[-1]
    if not isinstance(last, AIMessage) or last.tool_calls:
        return {"claim_check_needs_retry": False}

    retry_count = int(state.get("claim_check_retry_count", 0))
    max_claim_check_retries = compute_claim_check_retry_limit(runtime)
    max_empty_output_retries = compute_empty_output_retry_limit(runtime)
    log(
        state,
        "graph.claim_check.start",
        retry_count=retry_count,
        max_claim_check_retries=max_claim_check_retries,
        max_empty_output_retries=max_empty_output_retries,
    )

    assistant_text = _content_to_text(getattr(last, "content", "")).strip()
    if not assistant_text:
        if retry_count >= max_empty_output_retries:
            log(
                state,
                "graph.claim_check.empty_output_exhausted",
                retry_count=retry_count,
            )
            return {
                "claim_check_needs_retry": False,
                "claim_check_retry_count": 0,
            }
        backoff_seconds = _retry_backoff_seconds(retry_count)
        log(
            state,
            "graph.claim_check.empty_output_retry",
            retry_count=retry_count,
            backoff_seconds=backoff_seconds,
        )
        await asyncio.sleep(backoff_seconds)
        return {
            "messages": [
                SystemMessage(
                    content=(
                        "SELF_CHECK_EMPTY_OUTPUT: Your previous response produced no visible output. "
                        "Continue and provide a concrete answer or execute the needed tools now. "
                        "Do not stop silently."
                    )
                )
            ],
            "claim_check_needs_retry": True,
            "claim_check_retry_count": retry_count + 1,
        }

    turn_messages = _latest_turn_messages(messages)
    if not turn_messages:
        return {"claim_check_needs_retry": False}
    user_text = _content_to_text(getattr(turn_messages[0], "content", "")).strip()
    recent_tool_outputs = _collect_recent_tool_outputs(turn_messages)
    turn_window = _serialize_turn_window(turn_messages)
    tool_budget = max(
        300,
        min(3000, int(getattr(runtime, "_context_short_term_low_tokens", 3500) * 0.25)),
    )
    recent_tool_outputs = [
        _trim_text_to_token_budget(x, token_budget=tool_budget)
        for x in recent_tool_outputs[-8:]
    ]
    turn_window_budget = max(
        1200,
        min(6000, int(getattr(runtime, "_context_short_term_low_tokens", 3500) * 0.6)),
    )
    turn_window = _trim_text_to_token_budget(turn_window, token_budget=turn_window_budget)
    verdict = CompletionClaimVerification.from_any(
        await runtime.verify_completion_claim(
            user_text=user_text,
            assistant_text=assistant_text,
            recent_tool_outputs=recent_tool_outputs,
            turn_window=turn_window,
        )
    )
    log(
        state,
        "graph.claim_check.verdict",
        retry_count=retry_count,
        usable=bool(verdict.usable),
        mismatch=bool(verdict.mismatch),
        applies=bool(verdict.applies),
        confidence=verdict.confidence,
    )
    if not verdict.usable:
        if retry_count >= max_claim_check_retries:
            log(
                state,
                "graph.claim_check.unusable_exhausted",
                retry_count=retry_count,
            )
            return {"claim_check_needs_retry": False}
        backoff_seconds = _retry_backoff_seconds(retry_count)
        log(
            state,
            "graph.claim_check.unusable_retry",
            retry_count=retry_count,
            backoff_seconds=backoff_seconds,
        )
        await asyncio.sleep(backoff_seconds)
        reason = str(verdict.reason).strip()[:180]
        note = (
            "SELF_CHECK_UNAVAILABLE: Claim checker returned an unusable decision. "
            "Continue the turn and either execute the needed tool action now or restate status clearly."
        )
        if reason:
            note += f" Reason={reason}."
        return {
            "messages": [SystemMessage(content=note)],
            "claim_check_needs_retry": True,
            "claim_check_retry_count": retry_count + 1,
        }
    mismatch = bool(verdict.mismatch)
    if not mismatch:
        log(state, "graph.claim_check.passed", retry_count=retry_count)
        return {
            "claim_check_needs_retry": False,
            "claim_check_retry_count": 0,
        }
    if retry_count >= max_claim_check_retries:
        log(
            state,
            "graph.claim_check.mismatch_exhausted",
            retry_count=retry_count,
        )
        return {"claim_check_needs_retry": False}

    reason = str(verdict.reason).strip()[:180]
    repair = str(verdict.repair_instruction).strip()[:220]
    backoff_seconds = _retry_backoff_seconds(retry_count)
    log(
        state,
        "graph.claim_check.mismatch_retry",
        retry_count=retry_count,
        backoff_seconds=backoff_seconds,
        reason=reason,
    )
    await asyncio.sleep(backoff_seconds)
    note = (
        "SELF_CHECK_FAILED: Your last reply likely claimed an immediate action was done "
        "without confirmed tool evidence in this turn. "
        "Do not repeat the claim. Either execute the required tool now or state pending status clearly."
    )
    if reason:
        note += f" Reason={reason}."
    if repair:
        note += f" Fix={repair}."
    return {
        "messages": [SystemMessage(content=note)],
        "claim_check_needs_retry": True,
        "claim_check_retry_count": retry_count + 1,
    }
