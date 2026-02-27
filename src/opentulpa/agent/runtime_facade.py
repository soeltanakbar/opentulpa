"""Runtime API facade helpers for internal API and tool execution."""

from __future__ import annotations

from typing import Any

import httpx

from opentulpa.agent.runtime_tool_execution import (
    execute_tool as _execute_tool,
)


async def request_with_backoff(
    runtime: Any,
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    timeout: float = 20.0,
    retries: int = 2,
) -> httpx.Response:
    return await runtime._internal_api.request_with_backoff(
        method=method,
        path=path,
        params=params,
        json_body=json_body,
        timeout=timeout,
        retries=retries,
    )


async def execute_tool(
    runtime: Any,
    *,
    action_name: str,
    action_args: dict[str, Any],
    customer_id: str | None = None,
    inject_customer_id: bool = False,
    approval_execution_customer_id_tools: set[str],
) -> Any:
    return await _execute_tool(
        start_runtime=runtime.start,
        log_behavior_event=runtime.log_behavior_event,
        tools=runtime._tools,
        action_name=action_name,
        action_args=action_args,
        customer_id=customer_id,
        inject_customer_id=inject_customer_id,
        approval_execution_customer_id_tools=approval_execution_customer_id_tools,
        resolve_link_aliases_in_args=runtime.resolve_link_aliases_in_args,
        register_links_from_text=runtime.register_links_from_text,
    )
