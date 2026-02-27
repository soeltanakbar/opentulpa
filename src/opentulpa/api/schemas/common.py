"""Shared API schema base types."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class BaseRequestModel(BaseModel):
    """Default config for request DTOs."""

    model_config = ConfigDict(extra="ignore")
