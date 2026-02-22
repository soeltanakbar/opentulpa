"""Conversation-domain request/response contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ConversationTurnRequest:
    """A normalized user turn entering the application layer."""

    customer_id: str
    thread_id: str
    text: str
    include_pending_context: bool = True
    recursion_limit_override: int | None = None


@dataclass(slots=True)
class ConversationTurnResult:
    """A normalized assistant turn leaving the application layer."""

    customer_id: str
    thread_id: str
    text: str
    status: str = "ok"

