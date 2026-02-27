"""Tool execution graph node implementation."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from opentulpa.agent.lc_messages import AIMessage, ToolMessage
from opentulpa.agent.models import AgentState
from opentulpa.agent.utils import (
    content_to_text as _content_to_text,
)
from opentulpa.agent.utils import (
    extract_relative_delay_minutes as _extract_relative_delay_minutes,
)
from opentulpa.agent.utils import (
    is_cron_like_schedule as _is_cron_like_schedule,
)
from opentulpa.agent.utils import (
    latest_user_text as _latest_user_text,
)
from opentulpa.agent.utils import (
    safe_json as _safe_json,
)

CUSTOMER_ID_TOOLS: set[str] = {
    "memory_search",
    "memory_add",
    "uploaded_file_search",
    "uploaded_file_get",
    "uploaded_file_send",
    "tulpa_file_send",
    "web_image_send",
    "uploaded_file_analyze",
    "skill_list",
    "skill_get",
    "skill_upsert",
    "skill_delete",
    "directive_get",
    "directive_set",
    "directive_clear",
    "time_profile_get",
    "time_profile_set",
    "tulpa_run_terminal",
    "routine_list",
    "routine_create",
    "routine_delete",
    "automation_delete",
    "browser_use_run",
    "guardrail_execute_approved_action",
}


async def tools_node(
    state: AgentState,
    *,
    runtime: Any,
    log: Callable[..., None],
) -> dict[str, Any]:
    messages = state.get("messages", [])
    if not messages:
        return {}
    last = messages[-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return {}

    customer_id = state.get("customer_id", "")
    thread_id = str(state.get("thread_id", "")).strip()
    scheduled_origin = thread_id.lower().startswith("wake_") or thread_id.lower().startswith("wake-")
    log(
        state,
        "graph.tools.start",
        requested_tool_calls=len(last.tool_calls),
        execution_origin=("scheduled" if scheduled_origin else "interactive"),
    )

    latest_user_for_guard = _latest_user_text(messages)
    prior_assistant_for_guard = ""
    for msg in reversed(messages[:-1]):
        if isinstance(msg, AIMessage):
            candidate = _content_to_text(getattr(msg, "content", "")).strip()
            if candidate:
                prior_assistant_for_guard = candidate
                break

    tool_messages: list[ToolMessage] = []
    approval_handoff = False
    had_error = False
    for call in last.tool_calls:
        call_name = str(call.get("name", ""))
        call_id = str(call.get("id", ""))
        args = call.get("args", {}) or {}
        try:
            tool_fn = runtime._tools.get(call_name)
            if tool_fn is None:
                raise ValueError(f"Unknown tool: {call_name}")
            if call_name in CUSTOMER_ID_TOOLS:
                args = {**args, "customer_id": customer_id}
            if call_name in {"tulpa_run_terminal", "routine_create"}:
                args = {
                    **args,
                    "thread_id": thread_id,
                    "execution_origin": "scheduled" if scheduled_origin else "interactive",
                    "guard_context": {
                        "previous_user_message": latest_user_for_guard[:2000],
                        "previous_assistant_message": prior_assistant_for_guard[:2000],
                    },
                }
            if call_name == "routine_create":
                latest_user = _latest_user_text(messages)
                corrected_args = dict(args)
                delay_minutes = _extract_relative_delay_minutes(latest_user)
                if delay_minutes is not None and _is_cron_like_schedule(
                    str(corrected_args.get("schedule", ""))
                ):
                    run_at_local = datetime.now().astimezone() + timedelta(
                        minutes=max(1, delay_minutes)
                    )
                    corrected_args["schedule"] = run_at_local.isoformat()
                args = corrected_args
            args = runtime.resolve_link_aliases_in_args(customer_id=customer_id, args=args)
            result = await tool_fn.ainvoke(args)
            runtime.register_links_from_text(
                customer_id=customer_id,
                text=_safe_json(result),
                source=f"tool:{call_name}",
                limit=40,
            )
            result_text = _safe_json(result)
            log(
                state,
                "graph.tools.success",
                tool_name=call_name,
                tool_call_id=call_id,
                result_chars=len(result_text),
            )
            if (
                isinstance(result, dict)
                and str(result.get("status", "")).strip().lower() == "approval_pending"
                and str(result.get("approval_id", "")).strip()
            ):
                approval_handoff = True
                handoff_payload = {
                    "approval_id": str(result.get("approval_id", "")).strip(),
                    "summary": str(result.get("summary", "")).strip(),
                    "reason": str(result.get("reason", "")).strip(),
                    "action_name": call_name,
                }
                tool_messages.append(
                    ToolMessage(
                        content=f"APPROVAL_HANDOFF {json.dumps(handoff_payload, ensure_ascii=False)}",
                        tool_call_id=call_id,
                    )
                )
                log(
                    state,
                    "graph.tools.approval_handoff",
                    tool_name=call_name,
                    tool_call_id=call_id,
                )
            else:
                tool_messages.append(ToolMessage(content=_safe_json(result), tool_call_id=call_id))
        except Exception as exc:
            had_error = True
            log(
                state,
                "graph.tools.error",
                tool_name=call_name,
                tool_call_id=call_id,
                error=str(exc)[:500],
            )
            tool_messages.append(
                ToolMessage(
                    content=f"TOOL_ERROR: {call_name} failed: {exc}",
                    tool_call_id=call_id,
                )
            )
    update: dict[str, Any] = {"messages": tool_messages, "approval_handoff": approval_handoff}
    if had_error:
        update["tool_error_count"] = int(state.get("tool_error_count", 0)) + 1
        update["last_tool_error"] = "tool execution failed"
    log(
        state,
        "graph.tools.complete",
        emitted_messages=len(tool_messages),
        had_error=had_error,
    )
    return update
