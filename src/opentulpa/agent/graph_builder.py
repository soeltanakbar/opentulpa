"""Graph construction for OpenTulpa runtime."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from opentulpa.agent.lc_messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from opentulpa.agent.models import AgentState
from opentulpa.agent.utils import (
    approx_tokens as _approx_tokens,
)
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
    looks_like_shell_command as _looks_like_shell_command,
)
from opentulpa.agent.utils import (
    message_to_text as _message_to_text,
)
from opentulpa.agent.utils import (
    safe_json as _safe_json,
)


def _compute_claim_check_retry_limit(runtime: Any) -> int:
    """
    Derive retry budget from graph recursion limit so claim-check retries
    can keep driving progress instead of stopping after a tiny fixed count.
    """
    try:
        recursion_limit = int(getattr(runtime, "recursion_limit", 30))
    except Exception:
        recursion_limit = 30
    # Keep a small headroom for non-claim-check hops; still allow many retries.
    return max(3, min(24, recursion_limit - 6))


def _compute_empty_output_retry_limit(runtime: Any) -> int:
    """
    Empty assistant outputs should self-repair quickly and then exit.
    Long retry loops here burn context without producing user-visible progress.
    """
    return min(2, _compute_claim_check_retry_limit(runtime))


def build_runtime_graph(runtime: Any):
    assert runtime._model_with_tools is not None
    assert runtime._checkpointer is not None

    required_args: dict[str, tuple[str, ...]] = {
        "tulpa_write_file": ("path", "content"),
        "tulpa_validate_file": ("path",),
        "tulpa_read_file": ("path",),
        "tulpa_run_terminal": ("command",),
        "fetch_url_content": ("url",),
        "fetch_file_content": ("url",),
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
        "routine_create": (
            "name",
            "schedule",
            "message",
            "implementation_command",
            "customer_id",
        ),
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
            "routine_create with a local ISO timestamp schedule, notify_user=true, and a concrete "
            "implementation_command that will run at schedule time. "
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
            "1) If user gives a specific webpage URL to read, call fetch_url_content first. "
            "2) If user gives a direct file URL (pdf/docx/image), call fetch_file_content. "
            "3) If user needs general/current info discovery, use web_search first, then fetch_url_content/fetch_file_content for exact links. "
            "Never use legacy ':online' suffix models. "
            "4) Use browser_use_run only when the task requires real browser interaction (clicking, forms, login, "
            "multi-step navigation, dynamic JS rendering) or when simpler tools are insufficient. "
            "Prefer the cheapest/simplest tool path that can satisfy the objective. "
            "When a user provides a specific URL and asks to inspect/read/summarize it, "
            "call fetch_url_content or fetch_file_content based on URL type (do not rely only on web_search). "
            "When users refer to files they uploaded earlier (e.g. 'the table/orders file'), "
            "use uploaded_file_search, then uploaded_file_get/uploaded_file_analyze/uploaded_file_send as needed. "
            "If user asks to send/share a file back in this chat, call uploaded_file_send exactly once "
            "for the selected file and only state it was sent after a successful tool result. "
            "Use known link aliases (link_*) for very long URLs to reduce copy errors. "
            "If you output a known alias ID, it will be expanded to full URL for the user. "
            "When users share important links/files/IDs they may need later, proactively call memory_add and store exact URL/file name/file_id/path values. "
            "Files uploaded by users are persisted in the file vault (with summary/description); use uploaded_file_search/uploaded_file_get to retrieve records and memory_add exact file name/file_id/path when the user may need recall later. "
            "When you are unsure about prior user-specific facts/preferences/IDs because they are not in short-term context, "
            "call memory_search before asking the user to repeat themselves. "
            "Credential/token recovery policy: before asking users for keys, secrets, tokens, client files, or auth codes, "
            "first try memory_search and local-file lookup (tulpa_catalog/tulpa_read_file) for existing credentials. "
            "For OAuth integrations, prefer refresh-token recovery first; if a refresh token exists, refresh and persist "
            "updated token data, then retry the original task. "
            "Only ask the user for a new auth code when refresh-token recovery is impossible "
            "(missing/revoked refresh token or invalid client credentials). "
            "Background/routine execution policy: for recoverable errors (missing file/dependency/credential format mismatch), "
            "attempt a low-risk self-repair and retry before reporting failure. "
            "Never repeatedly ask for credentials that were already found in memory or local files. "
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
            "When a tool returns APPROVAL_PENDING, describe it as pending/requested only. "
            "Never say an action was already created/updated/deleted/executed in that same message. "
            "Guardrail model: approval checks happen at execution boundary tools (terminal/script execution and "
            "routine_create planning), not as a separate registration step. "
            "For routine_create, always provide a concrete implementation_command that describes planned "
            "scheduled behavior for guard evaluation. "
            "Scheduled/wake executions are pre-authorized and should run without per-run approval prompts. "
            "Never say an external action was sent/posted/executed until you have a successful tool result. "
            "If approval is still pending or execution was blocked, state clearly that it did not run yet. "
            "For capability requests, avoid generic marketing copy. "
            "Use a consultative format: "
            "1) concise capability overview grounded in actual tools, "
            "2) ask 2-3 diagnostic onboarding questions about user goals, bottlenecks, and desired integrations, "
            "3) propose one concrete next action. "
            "Do not claim completion while validation/tests are failing."
        )
    )

    def _log(state: AgentState | None, event: str, **fields: Any) -> None:
        log_event = getattr(runtime, "log_behavior_event", None)
        if not callable(log_event):
            return
        payload: dict[str, Any] = {}
        if isinstance(state, dict):
            trace_id = str(state.get("agent_trace_id", "")).strip()
            thread_id = str(state.get("thread_id", "")).strip()
            customer_id = str(state.get("customer_id", "")).strip()
            if trace_id:
                payload["trace_id"] = trace_id
            if thread_id:
                payload["thread_id"] = thread_id
            if customer_id:
                payload["customer_id"] = customer_id
        payload.update(fields)
        log_event(event=event, **payload)

    async def agent_node(state: AgentState) -> dict[str, Any]:
        customer_id = state.get("customer_id", "")
        thread_id = state.get("thread_id", "")
        messages = state.get("messages", [])
        latest_user = _latest_user_text(messages)
        _log(
            state,
            "graph.agent.start",
            message_count=len(messages),
            latest_user_chars=len(latest_user),
        )
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
        link_alias_context = runtime._build_link_alias_context(
            customer_id=customer_id,
            user_text=latest_user,
        )
        prompt_budget = max(4000, int(getattr(runtime, "_context_token_limit", 12000)))
        low_budget = max(1500, int(getattr(runtime, "_context_short_term_low_tokens", 3500)))
        optional_context_budget = max(1000, min(3600, int(low_budget * 0.7)))
        prompt_messages_base: list[AnyMessage] = [
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
        optional_messages: list[AnyMessage] = []
        if active_directive:
            directive_text = _trim_text_to_token_budget(
                active_directive,
                token_budget=max(120, min(420, int(low_budget * 0.12))),
            )
            if directive_text:
                optional_messages.append(
                    SystemMessage(
                        content=(
                            "Active persistent directive profile for this user. "
                            "Treat this as a high-priority preference unless user overrides it now:\n"
                            f"{directive_text}"
                        )
                    )
                )
        if thread_rollup:
            rollup_text = _trim_text_to_token_budget(
                thread_rollup,
                token_budget=max(300, min(1400, int(low_budget * 0.4))),
            )
            if rollup_text:
                optional_messages.append(
                    SystemMessage(
                        content=(
                            "Compressed older thread context (already summarized):\n"
                            f"{rollup_text}"
                        )
                    )
                )
        if skill_context:
            skill_text = _trim_text_to_token_budget(
                skill_context,
                token_budget=max(400, min(1800, int(low_budget * 0.45))),
            )
            if skill_text:
                optional_messages.append(
                    SystemMessage(
                        content=(
                            "Matched reusable skills for this user request "
                            f"(selected: {', '.join(skill_names) if skill_names else 'unknown'}):\n\n"
                            f"{skill_text}"
                        )
                    )
                )
        if link_alias_context:
            aliases_text = _trim_text_to_token_budget(
                link_alias_context,
                token_budget=max(120, min(320, int(low_budget * 0.08))),
            )
            if aliases_text:
                optional_messages.append(SystemMessage(content=aliases_text))

        prompt_messages: list[AnyMessage] = [*prompt_messages_base]
        if optional_messages:
            kept_optional: list[AnyMessage] = []
            used_optional_tokens = 0
            for msg in optional_messages:
                msg_tokens = _approx_tokens(_content_to_text(getattr(msg, "content", "")))
                if kept_optional and used_optional_tokens + msg_tokens > optional_context_budget:
                    continue
                kept_optional.append(msg)
                used_optional_tokens += msg_tokens
            prompt_messages.extend(kept_optional)
        prompt_overhead_tokens = sum(
            _approx_tokens(_content_to_text(getattr(msg, "content", "")))
            for msg in prompt_messages
        )
        max_overhead_tokens = max(1400, int(prompt_budget * 0.72))
        while len(prompt_messages) > len(prompt_messages_base) and prompt_overhead_tokens > max_overhead_tokens:
            prompt_messages.pop()
            prompt_overhead_tokens = sum(
                _approx_tokens(_content_to_text(getattr(msg, "content", "")))
                for msg in prompt_messages
            )
        history_budget = max(800, prompt_budget - prompt_overhead_tokens)
        bounded_messages = _tail_messages_to_token_budget(messages, token_budget=history_budget)
        _log(
            state,
            "graph.agent.prompt_ready",
            prompt_message_count=len(prompt_messages),
            prompt_overhead_tokens=prompt_overhead_tokens,
            history_budget=history_budget,
            history_message_count=len(bounded_messages),
            optional_context_messages=max(0, len(prompt_messages) - len(prompt_messages_base)),
        )
        response = await runtime._model_with_tools.ainvoke(
            [
                *prompt_messages,
                *bounded_messages,
            ]
        )
        response_text = _content_to_text(getattr(response, "content", ""))
        _log(
            state,
            "graph.agent.response",
            response_chars=len(response_text.strip()),
            tool_call_count=len(getattr(response, "tool_calls", []) or []),
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
        _log(
            state,
            "graph.validate_tools.start",
            tool_call_count=len(last.tool_calls),
        )

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
                if call_name == "routine_create" and "implementation_command" in missing:
                    validation_errors.append(
                        ToolMessage(
                            content=(
                                "ROUTINE_IMPLEMENTATION_COMMAND_REQUIRED: routine_create needs "
                                "implementation_command (a concrete shell/script command like "
                                "`python3 tulpa_stuff/scripts/digest.py`) describing what will run "
                                "on each scheduled execution. Repair the call and retry."
                            ),
                            tool_call_id=call_id,
                        )
                    )
                    continue
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
                implementation_command = str(args.get("implementation_command", "")).strip()
                if not implementation_command:
                    validation_errors.append(
                        ToolMessage(
                            content=(
                                "ROUTINE_IMPLEMENTATION_COMMAND_REQUIRED: routine_create must include "
                                "a non-empty implementation_command (shell/script command) so scheduled "
                                "runs execute a concrete implementation."
                            ),
                            tool_call_id=call_id,
                        )
                    )
                    continue
                if not _looks_like_shell_command(implementation_command):
                    validation_errors.append(
                        ToolMessage(
                            content=(
                                "ROUTINE_IMPLEMENTATION_COMMAND_INVALID: implementation_command must "
                                "be a concrete shell command (executable + args), not natural language."
                            ),
                            tool_call_id=call_id,
                        )
                    )
                    continue
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
            _log(
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
        _log(state, "graph.validate_tools.passed", tool_call_count=len(last.tool_calls))
        return {"tool_validation_passed": True}

    async def tools_node(state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return {}
        last = messages[-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {}

        customer_id = state.get("customer_id", "")
        thread_id = str(state.get("thread_id", "")).strip()
        scheduled_origin = thread_id.lower().startswith("wake_") or thread_id.lower().startswith("wake-")
        _log(
            state,
            "graph.tools.start",
            requested_tool_calls=len(last.tool_calls),
            execution_origin=("scheduled" if scheduled_origin else "interactive"),
        )
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
                }:
                    args = {**args, "customer_id": customer_id}
                if call_name in {"tulpa_run_terminal", "routine_create"}:
                    args = {
                        **args,
                        "thread_id": thread_id,
                        "execution_origin": "scheduled" if scheduled_origin else "interactive",
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
                _log(
                    state,
                    "graph.tools.success",
                    tool_name=call_name,
                    tool_call_id=call_id,
                    result_chars=len(result_text),
                )
                tool_messages.append(ToolMessage(content=_safe_json(result), tool_call_id=call_id))
            except Exception as exc:
                had_error = True
                _log(
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
        update: dict[str, Any] = {"messages": tool_messages}
        if had_error:
            update["tool_error_count"] = int(state.get("tool_error_count", 0)) + 1
            update["last_tool_error"] = "tool execution failed"
        _log(
            state,
            "graph.tools.complete",
            emitted_messages=len(tool_messages),
            had_error=had_error,
        )
        return update

    def _latest_turn_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
        if not messages:
            return []
        start = 0
        for idx in range(len(messages) - 1, -1, -1):
            if isinstance(messages[idx], HumanMessage):
                start = idx
                break
        return messages[start:]

    def _collect_recent_tool_outputs(turn_messages: list[AnyMessage]) -> list[str]:
        if not turn_messages:
            return []
        outputs: list[str] = []
        for msg in turn_messages:
            if isinstance(msg, ToolMessage):
                text = _content_to_text(getattr(msg, "content", "")).strip()
                if text:
                    outputs.append(text)
        return outputs

    def _serialize_turn_window(turn_messages: list[AnyMessage]) -> str:
        parts: list[str] = []
        for msg in turn_messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, ToolMessage):
                role = "tool"
            else:
                continue
            text = _content_to_text(getattr(msg, "content", "")).strip()
            if not text:
                continue
            chunk = f"[{role}] {text}"
            parts.append(chunk)
        return "\n".join(parts)

    def _trim_text_to_token_budget(text: str, *, token_budget: int) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        budget = max(1, int(token_budget))
        if _approx_tokens(raw) <= budget:
            return raw
        max_chars = max(800, budget * 4)
        if len(raw) <= max_chars:
            return raw
        reserve = max(64, max_chars // 2 - 8)
        compact = f"{raw[:reserve]}\n...\n{raw[-reserve:]}"
        while _approx_tokens(compact) > budget and reserve > 64:
            reserve = max(64, int(reserve * 0.85))
            compact = f"{raw[:reserve]}\n...\n{raw[-reserve:]}"
        return compact.strip()

    def _tail_messages_to_token_budget(
        all_messages: list[AnyMessage],
        *,
        token_budget: int,
    ) -> list[AnyMessage]:
        if not all_messages:
            return []
        budget = max(200, int(token_budget))
        kept_rev: list[AnyMessage] = []
        used = 0
        for msg in reversed(all_messages):
            text = _message_to_text(msg)
            tok = max(1, _approx_tokens(text))
            if kept_rev and used + tok > budget:
                break
            kept_rev.append(msg)
            used += tok
            if used >= budget:
                break
        kept_rev.reverse()
        return kept_rev

    async def claim_check_node(state: AgentState) -> dict[str, Any]:
        def _retry_backoff_seconds(retry_count: int) -> float:
            safe_retry = max(0, int(retry_count))
            return min(3.2, 0.2 * (2**safe_retry))

        messages = state.get("messages", [])
        if not messages:
            return {"claim_check_needs_retry": False}
        last = messages[-1]
        if not isinstance(last, AIMessage) or last.tool_calls:
            return {"claim_check_needs_retry": False}

        retry_count = int(state.get("claim_check_retry_count", 0))
        max_claim_check_retries = _compute_claim_check_retry_limit(runtime)
        max_empty_output_retries = _compute_empty_output_retry_limit(runtime)
        _log(
            state,
            "graph.claim_check.start",
            retry_count=retry_count,
            max_claim_check_retries=max_claim_check_retries,
            max_empty_output_retries=max_empty_output_retries,
        )

        assistant_text = _content_to_text(getattr(last, "content", "")).strip()
        if not assistant_text:
            if retry_count >= max_empty_output_retries:
                _log(
                    state,
                    "graph.claim_check.empty_output_exhausted",
                    retry_count=retry_count,
                )
                return {
                    "claim_check_needs_retry": False,
                    "claim_check_retry_count": 0,
                }
            backoff_seconds = _retry_backoff_seconds(retry_count)
            _log(
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
        verdict = await runtime.verify_completion_claim(
            user_text=user_text,
            assistant_text=assistant_text,
            recent_tool_outputs=recent_tool_outputs,
            turn_window=turn_window,
        )
        _log(
            state,
            "graph.claim_check.verdict",
            retry_count=retry_count,
            usable=bool(verdict.get("usable", True)),
            mismatch=bool(verdict.get("mismatch", False)),
            applies=bool(verdict.get("applies", False)),
            confidence=verdict.get("confidence"),
        )
        if not bool(verdict.get("usable", True)):
            if retry_count >= max_claim_check_retries:
                _log(
                    state,
                    "graph.claim_check.unusable_exhausted",
                    retry_count=retry_count,
                )
                return {"claim_check_needs_retry": False}
            backoff_seconds = _retry_backoff_seconds(retry_count)
            _log(
                state,
                "graph.claim_check.unusable_retry",
                retry_count=retry_count,
                backoff_seconds=backoff_seconds,
            )
            await asyncio.sleep(backoff_seconds)
            reason = str(verdict.get("reason", "")).strip()[:180]
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
        mismatch = bool(verdict.get("mismatch", False))
        if not mismatch:
            _log(state, "graph.claim_check.passed", retry_count=retry_count)
            return {
                "claim_check_needs_retry": False,
                "claim_check_retry_count": 0,
            }
        if retry_count >= max_claim_check_retries:
            _log(
                state,
                "graph.claim_check.mismatch_exhausted",
                retry_count=retry_count,
            )
            return {"claim_check_needs_retry": False}

        reason = str(verdict.get("reason", "")).strip()[:180]
        repair = str(verdict.get("repair_instruction", "")).strip()[:220]
        backoff_seconds = _retry_backoff_seconds(retry_count)
        _log(
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

    def route_after_agent(state: AgentState) -> Literal["validate_tools", "claim_check"]:
        messages = state.get("messages", [])
        if not messages:
            return "claim_check"
        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "validate_tools"
        return "claim_check"

    def route_after_validate(state: AgentState) -> Literal["tools", "agent"]:
        if state.get("tool_validation_passed", True):
            return "tools"
        return "agent"

    def route_after_claim_check(state: AgentState) -> Literal["agent", END]:
        if bool(state.get("claim_check_needs_retry", False)):
            return "agent"
        return END

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node, retry_policy=RetryPolicy(max_attempts=3))
    builder.add_node(
        "validate_tools",
        validate_tool_calls_node,
        retry_policy=RetryPolicy(max_attempts=2),
    )
    builder.add_node("tools", tools_node, retry_policy=RetryPolicy(max_attempts=2))
    builder.add_node("claim_check", claim_check_node, retry_policy=RetryPolicy(max_attempts=2))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, ["validate_tools", "claim_check"])
    builder.add_conditional_edges("validate_tools", route_after_validate, ["tools", "agent"])
    builder.add_edge("tools", "agent")
    builder.add_conditional_edges("claim_check", route_after_claim_check, ["agent", END])
    return builder.compile(checkpointer=runtime._checkpointer)
