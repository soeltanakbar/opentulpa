"""Graph construction for OpenTulpa runtime."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Literal

from langchain.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from opentulpa.agent.models import AgentState
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
    looks_like_shell_command as _looks_like_shell_command,
)
from opentulpa.agent.utils import (
    safe_json as _safe_json,
)


def build_runtime_graph(runtime: Any):
    assert runtime._model_with_tools is not None
    assert runtime._checkpointer is not None

    required_args: dict[str, tuple[str, ...]] = {
        "tulpa_write_file": ("path", "content"),
        "tulpa_validate_file": ("path",),
        "tulpa_read_file": ("path",),
        "tulpa_run_terminal": ("command",),
        "fetch_link_content": ("url",),
        "uploaded_file_search": ("query",),
        "uploaded_file_get": ("file_id",),
        "uploaded_file_send": ("file_id",),
        "uploaded_file_analyze": ("file_id",),
        "skill_get": ("name",),
        "skill_upsert": ("name", "description", "instructions"),
        "skill_delete": ("name",),
        "directive_set": ("directive",),
        "time_profile_set": ("utc_offset",),
    }

    system_prompt = SystemMessage(
        content=(
            "You are OpenTulpa. Use tools when needed. "
            "Always validate required tool arguments before calling. "
            "If a tool fails, self-repair once with a low-risk correction and retry. "
            "Do not output vague preambles; give concrete updates. "
            "When a user gives persistent behavior preferences (style, coding approach, process), "
            "call directive_set with a concise durable directive summary before your response. "
            "If the user asks to forget/reset old preferences, call directive_clear first. "
            "If the user asks what directive is active, call directive_get. "
            "For one-time reminders ('in X minutes/hours' or 'at <time> send me ...'), use "
            "routine_create with a local ISO timestamp schedule and notify_user=true. "
            "Do not manually convert one-time reminders into UTC cron expressions. "
            "For scheduled routines, default notify_user=true unless the user explicitly asks "
            "for no notifications/alerts. "
            "If user tells you their timezone or UTC offset, call time_profile_set. "
            "When a user provides a specific URL and asks to inspect/read/summarize it, "
            "call fetch_link_content first (do not rely only on web_search). "
            "When users refer to files they uploaded earlier (e.g. 'the table/orders file'), "
            "use uploaded_file_search, then uploaded_file_get/uploaded_file_analyze/uploaded_file_send as needed. "
            "When a user requests recurring behavior/workflow/persona, collaborate briefly to clarify and then "
            "store it as a reusable user skill via skill_upsert. "
            "On future related requests, use skill_list/skill_get if needed and follow matched skill guidance. "
            "When creating or editing code with tulpa_write_file, call tulpa_validate_file on each edited file. "
            "Before claiming code tasks are complete, run tulpa_run_terminal quality checks "
            "(at least ruff + compileall; run pytest when tests exist). "
            "Do not claim completion while validation/tests are failing."
        )
    )

    async def agent_node(state: AgentState) -> dict[str, Any]:
        customer_id = state.get("customer_id", "")
        thread_id = state.get("thread_id", "")
        messages = state.get("messages", [])
        latest_user = _latest_user_text(messages)
        cached_query = str(state.get("active_skill_query", "")).strip()
        cached_context = str(state.get("active_skill_context", "")).strip()
        cached_names = state.get("active_skill_names", []) or []
        skill_query = cached_query
        skill_context = cached_context
        skill_names = cached_names if isinstance(cached_names, list) else []
        if latest_user and latest_user != cached_query:
            resolved = await runtime._resolve_skill_context(customer_id, latest_user)
            skill_context = str(resolved.get("context", "")).strip()
            names = resolved.get("skill_names", [])
            skill_names = [str(n).strip() for n in names if str(n).strip()] if isinstance(names, list) else []
            skill_query = latest_user
        active_directive = await runtime._load_active_directive(customer_id)
        thread_rollup = runtime._load_thread_rollup(thread_id)
        live_time = await runtime._build_live_time_context(customer_id)
        prompt_messages: list[AnyMessage] = [
            system_prompt,
            SystemMessage(
                content=(
                    f"customer_id={customer_id}. "
                    "For memory/directive tools pass this as customer_id."
                )
            ),
            SystemMessage(
                content=(
                    "Live time context (auto-injected this turn):\n"
                    f"- server_time_local_iso: {live_time['server_time_local_iso']}\n"
                    f"- server_time_utc_iso: {live_time['server_time_utc_iso']}\n"
                    f"- server_utc_offset: {live_time['server_utc_offset']}\n"
                    f"- user_time_local_iso: {live_time['user_time_local_iso']}\n"
                    f"- user_utc_offset: {live_time['user_utc_offset']}\n"
                    f"- user_time_source: {live_time['user_time_source']}\n"
                    "Use these concrete values for all relative-time reasoning in this turn."
                )
            ),
        ]
        if active_directive:
            prompt_messages.append(
                SystemMessage(
                    content=(
                        "Active persistent directive profile for this user. "
                        "Treat this as a high-priority preference unless user overrides it now:\n"
                        f"{active_directive}"
                    )
                )
            )
        if thread_rollup:
            prompt_messages.append(
                SystemMessage(
                    content=(
                        "Compressed older thread context (already summarized):\n"
                        f"{thread_rollup}"
                    )
                )
            )
        if skill_context:
            prompt_messages.append(
                SystemMessage(
                    content=(
                        "Matched reusable skills for this user request "
                        f"(selected: {', '.join(skill_names) if skill_names else 'unknown'}):\n\n"
                        f"{skill_context}"
                    )
                )
            )
        response = await runtime._model_with_tools.ainvoke(
            [
                *prompt_messages,
                *messages,
            ]
        )
        update: dict[str, Any] = {"messages": [response]}
        if skill_query:
            update["active_skill_query"] = skill_query
            update["active_skill_context"] = skill_context
            update["active_skill_names"] = skill_names
        return update

    async def validate_tool_calls_node(state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return {"tool_validation_passed": True}
        last = messages[-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {"tool_validation_passed": True}

        validation_errors: list[ToolMessage] = []
        for call in last.tool_calls:
            call_name = str(call.get("name", ""))
            call_id = str(call.get("id", ""))
            args = call.get("args", {}) or {}
            if not isinstance(args, dict):
                validation_errors.append(
                    ToolMessage(
                        content=f"TOOL_VALIDATION_ERROR: arguments for {call_name} must be an object",
                        tool_call_id=call_id,
                    )
                )
                continue
            missing = [arg for arg in required_args.get(call_name, ()) if not args.get(arg)]
            if missing:
                validation_errors.append(
                    ToolMessage(
                        content=(
                            f"TOOL_VALIDATION_ERROR: missing required argument(s) for "
                            f"{call_name}: {', '.join(missing)}"
                        ),
                        tool_call_id=call_id,
                    )
                )
                continue
            if call_name == "tulpa_run_terminal":
                command = str(args.get("command", "")).strip()
                if not _looks_like_shell_command(command):
                    validation_errors.append(
                        ToolMessage(
                            content=(
                                "TOOL_VALIDATION_ERROR: command must be a concrete shell command "
                                "with executable + args."
                            ),
                            tool_call_id=call_id,
                        )
                    )
            if call_name == "routine_create":
                latest_user = _latest_user_text(messages)
                schedule = str(args.get("schedule", "")).strip()
                delay_minutes = _extract_relative_delay_minutes(latest_user)
                if delay_minutes is not None and _is_cron_like_schedule(schedule):
                    validation_errors.append(
                        ToolMessage(
                            content=(
                                "TOOL_VALIDATION_ERROR: for one-time relative reminders, "
                                "use a local ISO datetime schedule (not cron)."
                            ),
                            tool_call_id=call_id,
                        )
                    )
        if validation_errors:
            return {
                "messages": validation_errors,
                "tool_validation_passed": False,
                "tool_error_count": int(state.get("tool_error_count", 0)) + 1,
                "last_tool_error": "tool validation failed",
            }
        return {"tool_validation_passed": True}

    async def tools_node(state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return {}
        last = messages[-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {}

        customer_id = state.get("customer_id", "")
        tool_messages: list[ToolMessage] = []
        had_error = False
        for call in last.tool_calls:
            call_name = str(call.get("name", ""))
            call_id = str(call.get("id", ""))
            args = call.get("args", {}) or {}
            try:
                tool_fn = runtime._tools.get(call_name)
                if tool_fn is None:
                    raise ValueError(f"Unknown tool: {call_name}")
                if call_name in {
                    "memory_search",
                    "memory_add",
                    "uploaded_file_search",
                    "uploaded_file_get",
                    "uploaded_file_send",
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
                    "routine_create",
                }:
                    args = {**args, "customer_id": customer_id}
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
                result = await tool_fn.ainvoke(args)
                tool_messages.append(ToolMessage(content=_safe_json(result), tool_call_id=call_id))
            except Exception as exc:
                had_error = True
                tool_messages.append(
                    ToolMessage(
                        content=f"TOOL_ERROR: {call_name} failed: {exc}",
                        tool_call_id=call_id,
                    )
                )
        update: dict[str, Any] = {"messages": tool_messages}
        if had_error:
            update["tool_error_count"] = int(state.get("tool_error_count", 0)) + 1
            update["last_tool_error"] = "tool execution failed"
        return update

    def route_after_agent(state: AgentState) -> Literal["validate_tools", END]:
        messages = state.get("messages", [])
        if not messages:
            return END
        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "validate_tools"
        return END

    def route_after_validate(state: AgentState) -> Literal["tools", "agent"]:
        if state.get("tool_validation_passed", True):
            return "tools"
        return "agent"

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node, retry_policy=RetryPolicy(max_attempts=3))
    builder.add_node(
        "validate_tools",
        validate_tool_calls_node,
        retry_policy=RetryPolicy(max_attempts=2),
    )
    builder.add_node("tools", tools_node, retry_policy=RetryPolicy(max_attempts=2))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, ["validate_tools", END])
    builder.add_conditional_edges("validate_tools", route_after_validate, ["tools", "agent"])
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=runtime._checkpointer)
