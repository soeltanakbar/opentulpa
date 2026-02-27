"""Tool execution orchestration helper for runtime."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any


async def execute_tool(
    *,
    start_runtime: Callable[[], Awaitable[None]],
    log_behavior_event: Callable[..., None],
    tools: dict[str, Any],
    action_name: str,
    action_args: dict[str, Any],
    customer_id: str | None,
    inject_customer_id: bool,
    approval_execution_customer_id_tools: set[str],
    resolve_link_aliases_in_args: Callable[..., dict[str, Any]],
    register_links_from_text: Callable[..., list[dict[str, Any]]],
) -> Any:
    """
    Public runtime API for tool execution outside normal graph turns.

    Used by approval execution to avoid coupling to private runtime attributes.
    """
    await start_runtime()
    safe_action = str(action_name or "").strip()
    safe_customer = str(customer_id or "").strip()
    log_behavior_event(
        event="tool_execute_start",
        action_name=safe_action,
        customer_id=safe_customer,
    )
    tool_fn = tools.get(safe_action)
    if tool_fn is None:
        log_behavior_event(
            event="tool_execute_missing",
            action_name=safe_action,
            customer_id=safe_customer,
        )
        raise RuntimeError(f"unknown tool: {action_name}")
    args = action_args if isinstance(action_args, dict) else {}
    if inject_customer_id and safe_action in approval_execution_customer_id_tools:
        args = {**args, "customer_id": safe_customer}
    args = resolve_link_aliases_in_args(
        customer_id=safe_customer,
        args=args,
    )
    try:
        result = await tool_fn.ainvoke(args)
    except Exception as exc:
        log_behavior_event(
            event="tool_execute_error",
            action_name=safe_action,
            customer_id=safe_customer,
            error=str(exc)[:500],
        )
        raise
    if safe_customer:
        register_links_from_text(
            customer_id=safe_customer,
            text=json.dumps(result, ensure_ascii=False, default=str),
            source=f"tool:{safe_action}",
            limit=40,
        )
    log_behavior_event(
        event="tool_execute_complete",
        action_name=safe_action,
        customer_id=safe_customer,
        result_ok=(not isinstance(result, dict) or bool(result.get("ok", True))),
    )
    return result
