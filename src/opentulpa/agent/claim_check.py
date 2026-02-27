"""Claim-check helper functions for graph validation loops."""

from __future__ import annotations

from opentulpa.agent.lc_messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from opentulpa.agent.utils import (
    approx_tokens as _approx_tokens,
)
from opentulpa.agent.utils import (
    content_to_text as _content_to_text,
)
from opentulpa.agent.utils import (
    message_to_text as _message_to_text,
)


def latest_turn_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    if not messages:
        return []
    start = 0
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], HumanMessage):
            start = idx
            break
    return messages[start:]


def collect_recent_tool_outputs(turn_messages: list[AnyMessage]) -> list[str]:
    if not turn_messages:
        return []
    outputs: list[str] = []
    for msg in turn_messages:
        if isinstance(msg, ToolMessage):
            text = _content_to_text(getattr(msg, "content", "")).strip()
            if text:
                outputs.append(text)
    return outputs


def serialize_turn_window(turn_messages: list[AnyMessage]) -> str:
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
        parts.append(f"[{role}] {text}")
    return "\n".join(parts)


def trim_text_to_token_budget(text: str, *, token_budget: int) -> str:
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


def tail_messages_to_token_budget(
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


def retry_backoff_seconds(retry_count: int) -> float:
    safe_retry = max(0, int(retry_count))
    return min(3.2, 0.2 * (2**safe_retry))
