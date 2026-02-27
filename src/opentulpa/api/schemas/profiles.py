"""Schemas for profile routes."""

from __future__ import annotations

from opentulpa.api.schemas.common import BaseRequestModel


class DirectiveGetRequest(BaseRequestModel):
    customer_id: str = ""


class DirectiveSetRequest(BaseRequestModel):
    customer_id: str = ""
    directive: str = ""
    source: str = "agent"


class DirectiveClearRequest(BaseRequestModel):
    customer_id: str = ""


class TimeProfileGetRequest(BaseRequestModel):
    customer_id: str = ""


class TimeProfileSetRequest(BaseRequestModel):
    customer_id: str = ""
    utc_offset: str = ""
    source: str = "agent"
