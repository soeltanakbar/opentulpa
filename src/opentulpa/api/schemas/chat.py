"""Schemas for chat routes."""

from __future__ import annotations

from opentulpa.api.schemas.common import BaseRequestModel


class InternalChatRequest(BaseRequestModel):
    customer_id: str = ""
    text: str = ""
    thread_id: str = ""
    include_pending_context: bool = True
    recursion_limit_override: int | None = None
