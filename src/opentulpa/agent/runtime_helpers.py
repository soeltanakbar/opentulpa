"""Pure helper functions used by the agent runtime."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from opentulpa.agent.lc_messages import ToolMessage
from opentulpa.agent.utils import content_to_text as _content_to_text

_LINK_ID_TOKEN_RE = re.compile(r"\blink_[A-Za-z0-9]{4,12}\b")


def extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def strip_internal_json_prefix(text: str) -> str:
    """
    Remove internal control JSON prefixes that can leak into streamed user output.

    Example internal payloads:
    - {"selected": []} from skill selector
    - {"notify_user": false, "reason": "..."} from wake classifier
    """
    raw = str(text or "")
    working = raw.lstrip()
    changed = False
    decoder = json.JSONDecoder()

    while working.startswith("{"):
        try:
            parsed, end_idx = decoder.raw_decode(working)
        except Exception:
            break

        is_internal_selector = (
            isinstance(parsed, dict)
            and set(parsed.keys()) == {"selected"}
            and isinstance(parsed.get("selected"), list)
        )
        is_internal_classifier = (
            isinstance(parsed, dict)
            and "notify_user" in parsed
            and set(parsed.keys()).issubset({"notify_user", "reason"})
        )
        if not (is_internal_selector or is_internal_classifier):
            break

        working = working[end_idx:].lstrip()
        changed = True

    return working if changed else raw


def has_incomplete_internal_json_prefix(text: str) -> bool:
    """
    Detect incomplete internal control JSON prefixes during streaming.
    This prevents leaking partial internal payloads (e.g. '{"selected": ...') to users.
    """
    working = str(text or "").lstrip()
    if not working.startswith("{"):
        return False
    head = working[:400]
    if '"selected"' not in head and '"notify_user"' not in head:
        return False
    decoder = json.JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(working)
    except Exception:
        return True
    is_internal_selector = (
        isinstance(parsed, dict)
        and set(parsed.keys()) == {"selected"}
        and isinstance(parsed.get("selected"), list)
    )
    is_internal_classifier = (
        isinstance(parsed, dict)
        and "notify_user" in parsed
        and set(parsed.keys()).issubset({"notify_user", "reason"})
    )
    return bool(is_internal_selector or is_internal_classifier)


def format_pending_context(events: list[dict[str, Any]], *, payload_limit: int = 800) -> str:
    lines: list[str] = []
    for idx, event in enumerate(events, start=1):
        source = str(event.get("source", "event"))
        event_type = str(event.get("event_type", "update"))
        payload = event.get("payload", {})
        if isinstance(payload, dict):
            payload_text = json.dumps(payload, ensure_ascii=False)
        else:
            payload_text = str(payload)
        payload_text = " ".join(payload_text.split())
        if len(payload_text) > payload_limit:
            payload_text = payload_text[:payload_limit] + "..."
        lines.append(f"{idx}. [{source}/{event_type}] {payload_text}")
    return "\n".join(lines)


def extract_approval_handoff_payload(messages: list[Any]) -> dict[str, Any]:
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        raw_text = _content_to_text(getattr(msg, "content", "")).strip()
        if not raw_text or not raw_text.upper().startswith("APPROVAL_HANDOFF"):
            continue
        payload_text = raw_text[len("APPROVAL_HANDOFF") :].strip()
        if payload_text.startswith(":"):
            payload_text = payload_text[1:].strip()
        if not payload_text:
            return {}
        with suppress(Exception):
            parsed = json.loads(payload_text)
            if isinstance(parsed, dict):
                return parsed
        return {}
    return {}


def format_approval_handoff_reply(payload: dict[str, Any]) -> str:
    approval_id = str(payload.get("approval_id", "")).strip()
    summary = str(payload.get("summary", "")).strip()
    reason = str(payload.get("reason", "")).strip()
    if approval_id:
        return f"Approval required before external write. approval_id={approval_id}"
    if summary and reason:
        return f"Approval required before external write. summary={summary}; reason={reason}"
    return "Approval required before external write."


def resolve_link_aliases_in_args(
    *,
    args: dict[str, Any],
    expand_alias_text: Callable[[str], str],
) -> dict[str, Any]:
    if not isinstance(args, dict):
        return {}

    def _walk(value: Any) -> Any:
        if isinstance(value, str):
            if _LINK_ID_TOKEN_RE.search(value):
                return expand_alias_text(value)
            return value
        if isinstance(value, list):
            return [_walk(item) for item in value]
        if isinstance(value, dict):
            return {str(k): _walk(v) for k, v in value.items()}
        return value

    return {str(k): _walk(v) for k, v in args.items()}
