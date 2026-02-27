"""Schemas for Slack routes."""

from __future__ import annotations

from pydantic import Field

from opentulpa.api.schemas.common import BaseRequestModel


class SlackConsentRequest(BaseRequestModel):
    customer_id: str | None = None
    scope: str = "write"


class SlackPostRequest(BaseRequestModel):
    customer_id: str | None = None
    channel_id: str = ""
    text: str = ""
    thread_ts: str | None = None


class SlackChannelsQuery(BaseRequestModel):
    limit: int = Field(default=100, ge=1, le=1000)
    cursor: str = ""


class SlackChannelHistoryQuery(BaseRequestModel):
    limit: int = Field(default=20, ge=1, le=1000)
    cursor: str = ""
