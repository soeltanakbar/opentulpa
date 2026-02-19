"""Graph construction for OpenTulpa runtime."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from opentulpa.agent.lc_messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
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
        "web_image_send": ("url",),
        "uploaded_file_analyze": ("file_id",),
        "skill_get": ("name",),
        "skill_upsert": ("name", "description", "instructions"),
        "skill_delete": ("name",),
        "directive_set": ("directive",),
        "time_profile_set": ("utc_offset",),
        "browser_use_run": ("task",),
        "browser_use_task_get": ("task_id",),
        "browser_use_task_control": ("task_id",),
        "routine_list": ("customer_id",),
        "routine_delete": ("routine_id", "customer_id"),
        "automation_delete": ("routine_id", "customer_id"),
        "guardrail_execute_approved_action": ("approval_id", "customer_id"),
    }

    system_prompt = SystemMessage(
        content=(
            "You are OpenTulpa. Use tools when needed. "
            "Always validate required tool arguments before calling. "
            "If a tool fails, self-repair once with a low-risk correction and retry. "
            "Do not output vague preambles; give concrete updates. "
            "Default to concise answers: keep responses short and direct unless the user asks for depth. "
            "When a user gives persistent behavior preferences (style, coding approach, process), "
            "call directive_set with a concise durable directive summary before your response. "
            "If the user asks to forget/reset old preferences, call directive_clear first. "
            "If the user asks what directive is active, call directive_get. "
            "For one-time reminders ('in X minutes/hours' or 'at <time> send me ...'), use "
            "routine_create with a local ISO timestamp schedule and notify_user=true. "
            "Do not manually convert one-time reminders into UTC cron expressions. "
            "For scheduled routines, default notify_user=true unless the user explicitly asks "
            "for no notifications/alerts. "
            "When users ask to stop/cancel/remove reminders or scheduled scripts, use "
            "routine_list first, reason over returned routines, then call routine_delete by routine_id. "
            "Only claim success "
            "after tool output confirms deletion/verified_removed. "
            "When users ask to delete an automation and its related script/files, use "
            "automation_delete by routine_id with delete_files=true. "
            "When creating automations that depend on generated files, include cleanup_paths "
            "in routine_create so later cleanup can be deterministic. "
            "If user tells you their timezone or UTC offset, call time_profile_set. "
            "Tool-selection policy: "
            "1) If user gives a specific URL to read, call fetch_link_content first. "
            "2) If user needs general/current info discovery, use web_search first, then fetch_link_content for exact links. "
            "3) Use browser_use_run only when the task requires real browser interaction (clicking, forms, login, "
            "multi-step navigation, dynamic JS rendering) or when simpler tools are insufficient. "
            "Prefer the cheapest/simplest tool path that can satisfy the objective. "
            "When a user provides a specific URL and asks to inspect/read/summarize it, "
            "call fetch_link_content first (do not rely only on web_search). "
            "When users refer to files they uploaded earlier (e.g. 'the table/orders file'), "
            "use uploaded_file_search, then uploaded_file_get/uploaded_file_analyze/uploaded_file_send as needed. "
            "When users ask to send an image from the web, use web_search to find candidate URLs, "
            "then call web_image_send (it validates URL content-type is image/* before sending). "
            "When a task needs real interactive browsing (dynamic pages, multi-step navigation, "
            "authenticated workflows, JavaScript-heavy sites), use browser_use_run with constrained "
            "allowed_domains and conservative max_steps/timeouts. "
            "If needed, use browser_use_task_get for progress and browser_use_task_control to stop/pause. "
            "Keep browser task follow-up compact: request step previews only when explicitly required. "
            "When a user requests recurring behavior/workflow/persona, collaborate briefly to clarify and then "
            "store it as a reusable user skill via skill_upsert. "
            "On future related requests, use skill_list/skill_get if needed and follow matched skill guidance. "
            "When creating or editing code with tulpa_write_file, call tulpa_validate_file on each edited file. "
            "Before claiming code tasks are complete, run tulpa_run_terminal quality checks "
            "(at least ruff + compileall; run pytest when tests exist). "
            "If the user asks what you can do/capabilities, include concrete integration capabilities: "
            "setting/storing API keys from Telegram setup flows, building new service integrations by writing code, "
            "scheduling periodic polling jobs, and producing change summaries/alerts from API or web data. "
            "If a tool returns APPROVAL_PENDING with an approval_id, ask the user to approve it and then call "
            "guardrail_execute_approved_action with that approval_id. "
            "For capability requests, avoid generic marketing copy. "
            "Use a consultative format: "
            "1) concise capability overview grounded in actual tools, "
            "2) ask 2-3 diagnostic onboarding questions about user goals, bottlenecks, and desired integrations, "
            "3) propose one concrete next action. "
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
        allowed_ids_raw = state.get("guardrail_allowed_call_ids", [])
        allowed_ids = (
            {str(item).strip() for item in allowed_ids_raw if str(item).strip()}
            if isinstance(allowed_ids_raw, list)
            else set()
        )
        deferred_feedback = state.get("guardrail_feedback_messages", [])
        deferred_messages: list[ToolMessage] = []
        if isinstance(deferred_feedback, list):
            for item in deferred_feedback:
                if isinstance(item, ToolMessage):
                    deferred_messages.append(item)
        tool_messages: list[ToolMessage] = []
        had_error = False
        for call in last.tool_calls:
            call_name = str(call.get("name", ""))
            call_id = str(call.get("id", ""))
            if allowed_ids and call_id not in allowed_ids:
                continue
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
                    "routine_list",
                    "routine_create",
                    "routine_delete",
                    "automation_delete",
                    "browser_use_run",
                    "guardrail_execute_approved_action",
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
        if deferred_messages:
            tool_messages.extend(deferred_messages)
        update: dict[str, Any] = {"messages": tool_messages}
        if deferred_messages:
            update["guardrail_feedback_messages"] = []
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

    async def guardrail_precheck_node(state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return {"guardrail_has_executable_calls": False, "guardrail_allowed_call_ids": []}
        last = messages[-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {"guardrail_has_executable_calls": False, "guardrail_allowed_call_ids": []}

        customer_id = str(state.get("customer_id", "")).strip()
        thread_id = str(state.get("thread_id", "")).strip()
        allowed_call_ids: list[str] = []
        gate_messages: list[ToolMessage] = []
        for call in last.tool_calls:
            call_name = str(call.get("name", "")).strip()
            call_id = str(call.get("id", "")).strip()
            args = call.get("args", {})
            safe_args = args if isinstance(args, dict) else {}
            try:
                result = await runtime.evaluate_tool_guardrail(
                    customer_id=customer_id,
                    thread_id=thread_id,
                    action_name=call_name,
                    action_args=safe_args,
                )
            except Exception as exc:
                result = {
                    "gate": "require_approval",
                    "reason": f"guardrail_error:{exc}",
                    "summary": f"execute {call_name}",
                }
            if not isinstance(result, dict):
                result = {
                    "gate": "require_approval",
                    "reason": "guardrail_invalid_result",
                    "summary": f"execute {call_name}",
                }
            gate = str(result.get("gate", "require_approval")).strip().lower()
            if gate == "allow":
                allowed_call_ids.append(call_id)
                continue
            approval_id = str(result.get("approval_id", "")).strip()
            summary = str(result.get("summary", f"execute {call_name}")).strip()
            reason = str(result.get("reason", "approval_required")).strip()
            if approval_id:
                content = (
                    "APPROVAL_PENDING: This action is blocked until user approval. "
                    f"approval_id={approval_id}; summary={summary}; reason={reason}. "
                    "After approval, call guardrail_execute_approved_action with this approval_id."
                )
            else:
                content = (
                    "TOOL_DENIED: guardrail denied action and no approval can be requested. "
                    f"summary={summary}; reason={reason}."
                )
            gate_messages.append(ToolMessage(content=content, tool_call_id=call_id))

        update: dict[str, Any] = {
            "guardrail_allowed_call_ids": allowed_call_ids,
            "guardrail_has_executable_calls": bool(allowed_call_ids),
        }
        if gate_messages:
            if allowed_call_ids:
                update["guardrail_feedback_messages"] = gate_messages
            else:
                update["messages"] = gate_messages
            update["tool_error_count"] = int(state.get("tool_error_count", 0)) + len(gate_messages)
            update["last_tool_error"] = "guardrail blocked one or more actions"
        return update

    def route_after_validate(state: AgentState) -> Literal["guardrail_precheck", "agent"]:
        if state.get("tool_validation_passed", True):
            return "guardrail_precheck"
        return "agent"

    def route_after_guardrail(state: AgentState) -> Literal["tools", "agent"]:
        if bool(state.get("guardrail_has_executable_calls", False)):
            return "tools"
        return "agent"

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node, retry_policy=RetryPolicy(max_attempts=3))
    builder.add_node(
        "validate_tools",
        validate_tool_calls_node,
        retry_policy=RetryPolicy(max_attempts=2),
    )
    builder.add_node(
        "guardrail_precheck",
        guardrail_precheck_node,
        retry_policy=RetryPolicy(max_attempts=2),
    )
    builder.add_node("tools", tools_node, retry_policy=RetryPolicy(max_attempts=2))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, ["validate_tools", END])
    builder.add_conditional_edges(
        "validate_tools", route_after_validate, ["guardrail_precheck", "agent"]
    )
    builder.add_conditional_edges("guardrail_precheck", route_after_guardrail, ["tools", "agent"])
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=runtime._checkpointer)
