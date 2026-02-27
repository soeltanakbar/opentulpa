"""Tool-call validation graph node implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opentulpa.agent.lc_messages import AIMessage, ToolMessage
from opentulpa.agent.models import AgentState
from opentulpa.agent.tool_call_validation import validate_tool_call as _validate_tool_call


async def validate_tool_calls_node(
    state: AgentState,
    *,
    log: Callable[..., None],
) -> dict[str, Any]:
    messages = state.get("messages", [])
    if not messages:
        return {"tool_validation_passed": True}
    last = messages[-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return {"tool_validation_passed": True}
    log(
        state,
        "graph.validate_tools.start",
        tool_call_count=len(last.tool_calls),
    )

    validation_errors: list[ToolMessage] = []
    for call in last.tool_calls:
        call_name = str(call.get("name", ""))
        call_id = str(call.get("id", ""))
        args = call.get("args", {}) or {}
        maybe_error = _validate_tool_call(
            call_name=call_name,
            call_id=call_id,
            args=args,
            messages=messages,
        )
        if maybe_error is not None:
            validation_errors.append(maybe_error)
    if validation_errors:
        log(
            state,
            "graph.validate_tools.failed",
            error_count=len(validation_errors),
        )
        return {
            "messages": validation_errors,
            "tool_validation_passed": False,
            "tool_error_count": int(state.get("tool_error_count", 0)) + 1,
            "last_tool_error": "tool validation failed",
        }
    log(state, "graph.validate_tools.passed", tool_call_count=len(last.tool_calls))
    return {"tool_validation_passed": True}
