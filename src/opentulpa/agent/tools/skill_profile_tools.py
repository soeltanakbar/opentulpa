"""Skill/profile LangChain tool bundle."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.tools import tool


def build_skill_profile_tools(
    *,
    runtime: Any,
    sync_proactive_heartbeat: Callable[[str, str], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    """Build skill and profile tools."""

    @tool
    async def skill_list(customer_id: str, include_global: bool = True, limit: int = 50) -> Any:
        """List reusable skills available to this user."""
        safe_limit = max(1, min(int(limit), 200))
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/skills/list",
            json_body={
                "customer_id": customer_id,
                "include_global": bool(include_global),
                "include_disabled": False,
                "limit": safe_limit,
            },
            timeout=8.0,
        )
        if r.status_code != 200:
            return {"error": f"skill_list failed: {r.text}"}
        return r.json().get("skills", [])

    @tool
    async def skill_get(
        name: str,
        customer_id: str,
        include_files: bool = True,
        include_global: bool = True,
    ) -> Any:
        """Get one skill by name, using user-scope first then global fallback."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/skills/get",
            json_body={
                "customer_id": customer_id,
                "name": name,
                "include_files": bool(include_files),
                "include_global": bool(include_global),
            },
            timeout=8.0,
        )
        if r.status_code != 200:
            return {"error": f"skill_get failed: {r.text}"}
        return r.json().get("skill", {})

    @tool
    async def skill_upsert(
        name: str,
        description: str,
        instructions: str,
        customer_id: str,
        scope: str = "user",
        supporting_files: dict[str, str] | None = None,
    ) -> Any:
        """Create or update a reusable skill for this user (or global when explicitly chosen)."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/skills/upsert",
            json_body={
                "customer_id": customer_id,
                "scope": scope,
                "name": name,
                "description": description,
                "instructions": instructions,
                "supporting_files": supporting_files if isinstance(supporting_files, dict) else None,
                "source": "langgraph_tool",
            },
            timeout=20.0,
        )
        if r.status_code != 200:
            return {"error": f"skill_upsert failed: {r.text}"}
        return r.json().get("skill", {})

    @tool
    async def skill_delete(name: str, customer_id: str, scope: str = "user") -> Any:
        """Delete a reusable skill by name."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/skills/delete",
            json_body={
                "customer_id": customer_id,
                "scope": scope,
                "name": name,
            },
            timeout=8.0,
        )
        if r.status_code != 200:
            return {"error": f"skill_delete failed: {r.text}"}
        return r.json()

    @tool
    async def directive_get(customer_id: str) -> Any:
        """Get the active persistent directive profile for this user."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/directive/get",
            json_body={"customer_id": customer_id},
            timeout=5.0,
        )
        if r.status_code != 200:
            return {"error": f"directive_get failed: {r.text}"}
        return r.json()

    @tool
    async def directive_set(directive: str, customer_id: str) -> Any:
        """Set or overwrite the user's persistent directive profile."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/directive/set",
            json_body={
                "customer_id": customer_id,
                "directive": directive,
                "source": "langgraph_tool",
            },
            timeout=5.0,
        )
        if r.status_code != 200:
            return {"error": f"directive_set failed: {r.text}"}
        payload = r.json()
        heartbeat = await sync_proactive_heartbeat(customer_id, directive)
        payload["proactive_heartbeat"] = heartbeat
        return payload

    @tool
    async def directive_clear(customer_id: str) -> Any:
        """Clear the user's persistent directive profile."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/directive/clear",
            json_body={"customer_id": customer_id},
            timeout=5.0,
        )
        if r.status_code != 200:
            return {"error": f"directive_clear failed: {r.text}"}
        payload = r.json()
        heartbeat = await sync_proactive_heartbeat(customer_id, "disable proactive mode")
        payload["proactive_heartbeat"] = heartbeat
        return payload

    @tool
    async def time_profile_get(customer_id: str) -> Any:
        """Get stored user UTC offset (if known)."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/time_profile/get",
            json_body={"customer_id": customer_id},
            timeout=5.0,
        )
        if r.status_code != 200:
            return {"error": f"time_profile_get failed: {r.text}"}
        return r.json()

    @tool
    async def time_profile_set(utc_offset: str, customer_id: str) -> Any:
        """Set user UTC offset in +HH:MM or -HH:MM format."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/time_profile/set",
            json_body={
                "customer_id": customer_id,
                "utc_offset": utc_offset,
                "source": "langgraph_tool",
            },
            timeout=5.0,
        )
        if r.status_code != 200:
            return {"error": f"time_profile_set failed: {r.text}"}
        return r.json()

    return {
        "skill_list": skill_list,
        "skill_get": skill_get,
        "skill_upsert": skill_upsert,
        "skill_delete": skill_delete,
        "directive_get": directive_get,
        "directive_set": directive_set,
        "directive_clear": directive_clear,
        "time_profile_get": time_profile_get,
        "time_profile_set": time_profile_set,
    }
