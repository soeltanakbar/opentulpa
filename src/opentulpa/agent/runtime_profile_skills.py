"""Skill/profile resolution helpers for the runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

import httpx

from opentulpa.agent.lc_messages import HumanMessage, SystemMessage
from opentulpa.agent.utils import content_to_text as _content_to_text


async def load_active_directive(
    *,
    customer_id: str,
    customer_profile_service: Any | None,
    request_with_backoff: Callable[..., Awaitable[httpx.Response]],
) -> str | None:
    cid = str(customer_id or "").strip()
    if not cid:
        return None
    if customer_profile_service is not None:
        try:
            return customer_profile_service.get_directive(cid)
        except Exception:
            pass
    try:
        r = await request_with_backoff(
            "POST",
            "/internal/directive/get",
            json_body={"customer_id": cid},
            timeout=5.0,
            retries=1,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        directive = str(data.get("directive") or "").strip()
        return directive or None
    except Exception:
        return None


async def load_user_utc_offset(
    *,
    customer_id: str,
    customer_profile_service: Any | None,
    request_with_backoff: Callable[..., Awaitable[httpx.Response]],
) -> str | None:
    cid = str(customer_id or "").strip()
    if not cid:
        return None
    if customer_profile_service is not None:
        with suppress(Exception):
            return customer_profile_service.get_utc_offset(cid)
    try:
        r = await request_with_backoff(
            "POST",
            "/internal/time_profile/get",
            json_body={"customer_id": cid},
            timeout=5.0,
            retries=1,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        offset = str(data.get("utc_offset") or "").strip()
        return offset or None
    except Exception:
        return None


async def list_available_skills(
    *,
    customer_id: str,
    request_with_backoff: Callable[..., Awaitable[httpx.Response]],
) -> list[dict[str, Any]]:
    cid = str(customer_id or "").strip()
    try:
        r = await request_with_backoff(
            "POST",
            "/internal/skills/list",
            json_body={
                "customer_id": cid,
                "include_global": True,
                "include_disabled": False,
                "limit": 200,
            },
            timeout=8.0,
            retries=1,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        skills = data.get("skills", [])
        if not isinstance(skills, list):
            return []
        out: list[dict[str, Any]] = []
        for item in skills:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            desc = str(item.get("description", "")).strip()
            scope = str(item.get("scope", "")).strip() or "user"
            if not name or not desc:
                continue
            out.append(
                {
                    "name": name,
                    "description": desc,
                    "scope": scope,
                }
            )
        return out
    except Exception:
        return []


async def select_relevant_skills(
    *,
    customer_id: str,
    query: str,
    candidates: list[dict[str, Any]],
    model: Any,
    extract_json_object: Callable[[str], dict[str, Any] | None],
    max_skills: int = 2,
) -> list[dict[str, Any]]:
    prompt_query = str(query or "").strip()
    if not prompt_query or not candidates:
        return []
    shortlist = candidates[:80]
    catalog = "\n".join(
        [
            f"- name={c['name']} scope={c['scope']} description={c['description'][:300]}"
            for c in shortlist
        ]
    )
    try:
        response = await model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You select reusable skills for the current user request.\n"
                        "Return strict JSON object with key 'selected', an array of objects:\n"
                        "  {\"name\": string, \"score\": number, \"reason\": string}\n"
                        "Choose only skills that materially improve answer quality.\n"
                        "If none apply, return {\"selected\": []}."
                    )
                ),
                HumanMessage(
                    content=(
                        f"customer_id={customer_id}\n"
                        f"user_request={prompt_query[:2000]}\n\n"
                        f"available_skills:\n{catalog}"
                    )
                ),
            ]
        )
        raw = _content_to_text(getattr(response, "content", "")).strip()
        parsed = extract_json_object(raw) or {}
        selected_raw = parsed.get("selected", [])
        if not isinstance(selected_raw, list):
            return []
        by_name = {c["name"]: c for c in shortlist}
        selected: list[dict[str, Any]] = []
        for item in selected_raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name or name not in by_name:
                continue
            score_raw = item.get("score", 0)
            try:
                score = float(score_raw)
            except Exception:
                score = 0.0
            if score < 0.45:
                continue
            selected.append(
                {
                    **by_name[name],
                    "score": score,
                    "reason": str(item.get("reason", "")).strip()[:300],
                }
            )
        selected.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return selected[: max(1, min(int(max_skills), 3))]
    except Exception:
        return []


async def resolve_skill_context(
    *,
    customer_id: str,
    user_text: str,
    model: Any,
    request_with_backoff: Callable[..., Awaitable[httpx.Response]],
    extract_json_object: Callable[[str], dict[str, Any] | None],
) -> dict[str, Any]:
    cid = str(customer_id or "").strip()
    query = str(user_text or "").strip()
    if not cid or not query:
        return {"skill_names": [], "context": ""}
    candidates = await list_available_skills(
        customer_id=cid,
        request_with_backoff=request_with_backoff,
    )
    if not candidates:
        return {"skill_names": [], "context": ""}
    selected = await select_relevant_skills(
        customer_id=cid,
        query=query,
        candidates=candidates,
        model=model,
        extract_json_object=extract_json_object,
        max_skills=1,
    )
    if not selected:
        return {"skill_names": [], "context": ""}

    sections: list[str] = []
    skill_names: list[str] = []
    total_chars = 0
    max_total_chars = 9000
    for item in selected:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        try:
            r = await request_with_backoff(
                "POST",
                "/internal/skills/get",
                json_body={
                    "customer_id": cid,
                    "name": name,
                    "include_files": False,
                    "include_global": True,
                },
                timeout=8.0,
                retries=1,
            )
            if r.status_code != 200:
                continue
            payload = r.json()
            skill = payload.get("skill", {})
            if not isinstance(skill, dict):
                continue
            skill_md = str(skill.get("skill_markdown", "")).strip()
            if not skill_md:
                continue
            snippet = (
                f"Skill name: {name}\n"
                f"Scope: {skill.get('scope', '')}\n"
                f"Description: {skill.get('description', '')}\n"
                f"Selection reason: {item.get('reason', '')}\n\n"
                f"SKILL.md:\n{skill_md[:3500]}"
            )
            if total_chars + len(snippet) > max_total_chars:
                break
            sections.append(snippet)
            skill_names.append(name)
            total_chars += len(snippet)
        except Exception:
            continue
    context = "\n\n---\n\n".join(sections).strip()
    return {"skill_names": skill_names, "context": context}


async def pre_resolve_skill_state(
    *,
    customer_id: str,
    user_text: str,
    model: Any,
    request_with_backoff: Callable[..., Awaitable[httpx.Response]],
    extract_json_object: Callable[[str], dict[str, Any] | None],
) -> dict[str, Any]:
    query = str(user_text or "").strip()
    if not query:
        return {
            "active_skill_query": "",
            "active_skill_context": "",
            "active_skill_names": [],
        }
    resolved = await resolve_skill_context(
        customer_id=customer_id,
        user_text=query,
        model=model,
        request_with_backoff=request_with_backoff,
        extract_json_object=extract_json_object,
    )
    context = str(resolved.get("context", "")).strip()
    names_raw = resolved.get("skill_names", [])
    names = [str(n).strip() for n in names_raw if str(n).strip()] if isinstance(names_raw, list) else []
    return {
        "active_skill_query": query,
        "active_skill_context": context,
        "active_skill_names": names,
    }
