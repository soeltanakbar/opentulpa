"""Agent graph node implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opentulpa.agent.claim_check import (
    tail_messages_to_token_budget as _tail_messages_to_token_budget,
)
from opentulpa.agent.claim_check import trim_text_to_token_budget as _trim_text_to_token_budget
from opentulpa.agent.lc_messages import (
    AnyMessage,
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
    latest_user_text as _latest_user_text,
)


async def agent_node(
    state: AgentState,
    *,
    runtime: Any,
    system_prompt: SystemMessage,
    log: Callable[..., None],
) -> dict[str, Any]:
    customer_id = state.get("customer_id", "")
    thread_id = state.get("thread_id", "")
    messages = state.get("messages", [])
    latest_user = _latest_user_text(messages)
    log(
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
    sanitized_history: list[AnyMessage] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_text = _content_to_text(getattr(msg, "content", "")).strip().lower()
            if (
                "approval_pending" in tool_text
                or "approval_handoff" in tool_text
                or '"status":"approval_pending"' in tool_text
                or '"status": "approval_pending"' in tool_text
            ):
                continue
        sanitized_history.append(msg)
    bounded_messages = _tail_messages_to_token_budget(
        sanitized_history,
        token_budget=history_budget,
    )
    log(
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
    log(
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
