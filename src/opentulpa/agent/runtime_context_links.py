"""Runtime helpers for pending context wrapping and link alias operations."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import Any


def prepend_pending_context(
    *,
    context_events: Any | None,
    customer_id: str,
    text: str,
    include_pending_context: bool,
    format_pending_context: Callable[[list[dict[str, Any]]], str],
) -> tuple[str, int | None]:
    if not include_pending_context or context_events is None:
        return text, None
    pending = context_events.list_events(customer_id, limit=20)
    if not pending:
        return text, None
    through_id = int(pending[-1]["id"])
    wrapped = (
        "System context updates collected while the user was away:\n"
        f"{format_pending_context(pending)}\n\n"
        f"User message:\n{text}"
    )
    return wrapped, through_id


def register_links_from_text(
    *,
    link_alias_service: Any | None,
    customer_id: str,
    text: str,
    source: str,
    limit: int = 40,
) -> list[dict[str, Any]]:
    if link_alias_service is None:
        return []
    cid = str(customer_id or "").strip()
    if not cid:
        return []
    raw = str(text or "")
    if not raw:
        return []
    with suppress(Exception):
        return link_alias_service.register_links_from_text(
            cid,
            raw,
            source=source,
            limit=limit,
        )
    return []


def expand_link_aliases(
    *,
    link_alias_service: Any | None,
    customer_id: str,
    text: str,
) -> str:
    if link_alias_service is None:
        return str(text or "")
    cid = str(customer_id or "").strip()
    raw = str(text or "")
    if not cid or not raw or "link_" not in raw.lower():
        return raw
    with suppress(Exception):
        return link_alias_service.expand_link_ids_in_text(cid, raw)
    return raw


def build_link_alias_context(
    *,
    link_alias_service: Any | None,
    customer_id: str,
    user_text: str,
) -> str:
    if link_alias_service is None:
        return ""
    cid = str(customer_id or "").strip()
    if not cid:
        return ""
    safe_user_text = str(user_text or "")
    seen_ids: set[str] = set()
    selected: list[dict[str, Any]] = []

    try:
        mentioned = link_alias_service.extract_link_ids(safe_user_text, limit=8)
    except Exception:
        mentioned = []
    for link_id in mentioned:
        with suppress(Exception):
            item = link_alias_service.get_by_id(cid, link_id)
            if not item:
                continue
            lid = str(item.get("id", "")).strip().lower()
            if not lid or lid in seen_ids:
                continue
            seen_ids.add(lid)
            selected.append(item)

    max_aliases = 4
    if len(selected) < max_aliases:
        recent: list[dict[str, Any]] = []
        with suppress(Exception):
            recent = link_alias_service.list_recent(cid, limit=max_aliases)
        for item in recent:
            lid = str(item.get("id", "")).strip().lower()
            if not lid or lid in seen_ids:
                continue
            seen_ids.add(lid)
            selected.append(item)
            if len(selected) >= max_aliases:
                break

    if not selected:
        return ""
    lines = [f"- {item['id']}: {item['url']}" for item in selected[:max_aliases]]
    return (
        "Known long-link aliases for this user:\n"
        + "\n".join(lines)
        + "\nUse alias IDs for long URLs. Outputting a known alias expands to the full URL."
    )
