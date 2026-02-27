"""Tool registration for the OpenTulpa LangGraph runtime."""

from __future__ import annotations

from typing import Any

from langchain.tools import tool

from opentulpa.agent.tools import (
    build_browser_use_tools,
    build_content_fetch_tools,
    build_file_tools,
    build_memory_tools,
    build_skill_profile_tools,
    build_tulpa_tools,
    build_workflow_tools,
)
from opentulpa.agent.tools_registry_support import (
    approval_pending_payload as _approval_pending_payload,
)
from opentulpa.agent.tools_registry_support import (
    browser_use_api_key as _browser_use_api_key,
)
from opentulpa.agent.tools_registry_support import (
    browser_use_base_url as _browser_use_base_url,
)
from opentulpa.agent.tools_registry_support import (
    browser_use_error_detail as _browser_use_error_detail,
)
from opentulpa.agent.tools_registry_support import (
    compact_browser_use_task_view as _compact_browser_use_task_view,
)
from opentulpa.agent.tools_registry_support import (
    normalize_allowed_domains as _normalize_allowed_domains,
)
from opentulpa.agent.tools_registry_support import (
    normalize_cleanup_paths as _normalize_cleanup_paths,
)
from opentulpa.agent.tools_registry_support import (
    normalize_execution_origin as _normalize_execution_origin,
)
from opentulpa.agent.tools_registry_support import (
    sync_proactive_heartbeat as _sync_proactive_heartbeat,
)
from opentulpa.agent.utils import looks_like_shell_command as _looks_like_shell_command
from opentulpa.policy.execution_boundary import ExecutionBoundaryGuard


def register_runtime_tools(runtime: Any) -> dict[str, Any]:
    boundary_guard = ExecutionBoundaryGuard(runtime=runtime)

    memory_tools = build_memory_tools(runtime=runtime)
    file_tools = build_file_tools(runtime=runtime)
    skill_profile_tools = build_skill_profile_tools(
        runtime=runtime,
        sync_proactive_heartbeat=lambda customer_id, directive_text: _sync_proactive_heartbeat(
            runtime=runtime,
            customer_id=customer_id,
            directive_text=directive_text,
        ),
    )
    browser_use_tools = build_browser_use_tools(
        runtime=runtime,
        browser_use_api_key=_browser_use_api_key,
        browser_use_base_url=_browser_use_base_url,
        browser_use_error_detail=_browser_use_error_detail,
        normalize_allowed_domains=_normalize_allowed_domains,
        compact_browser_use_task_view=_compact_browser_use_task_view,
    )
    content_fetch_tools = build_content_fetch_tools(runtime=runtime)
    tulpa_tools = build_tulpa_tools(
        runtime=runtime,
        boundary_guard=boundary_guard,
        normalize_execution_origin=_normalize_execution_origin,
        approval_pending_payload=_approval_pending_payload,
        looks_like_shell_command=_looks_like_shell_command,
    )
    workflow_tools = build_workflow_tools(
        runtime=runtime,
        boundary_guard=boundary_guard,
        normalize_execution_origin=_normalize_execution_origin,
        approval_pending_payload=_approval_pending_payload,
        normalize_cleanup_paths=_normalize_cleanup_paths,
        looks_like_shell_command=_looks_like_shell_command,
    )

    @tool
    async def web_search(query: str) -> Any:
        """Search the web for current information."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/web_search",
            json_body={"query": query},
            timeout=90.0,
        )
        if r.status_code != 200:
            return {"error": "web_search request failed"}
        return r.json().get("result", "No result.")

    return {
        **memory_tools,
        **file_tools,
        **skill_profile_tools,
        "web_search": web_search,
        **browser_use_tools,
        **content_fetch_tools,
        **tulpa_tools,
        **workflow_tools,
    }
