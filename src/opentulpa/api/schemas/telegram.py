"""Schemas for Telegram webhook routes."""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict

from opentulpa.api.schemas.common import BaseRequestModel


class TelegramWebhookRequest(BaseRequestModel):
    """Typed envelope for Telegram webhook updates.

    Top-level extras are preserved because Telegram update variants evolve
    and may include keys this server does not model directly.
    """

    model_config = ConfigDict(extra="allow")

    update_id: int | None = None
    message: dict[str, Any] | None = None
    edited_message: dict[str, Any] | None = None
    callback_query: dict[str, Any] | None = None
