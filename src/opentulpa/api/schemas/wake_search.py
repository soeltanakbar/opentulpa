"""Schemas for wake and web-search routes."""

from __future__ import annotations

from typing import Any

from pydantic import RootModel

from opentulpa.api.schemas.common import BaseRequestModel


class WakePayload(RootModel[dict[str, Any]]):
    pass


class WebSearchRequest(BaseRequestModel):
    query: str = ""
