"""Schemas for tulpa routes."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from opentulpa.api.schemas.common import BaseRequestModel


class TulpaWriteFileRequest(BaseRequestModel):
    path: str = ""
    content: Any | None = None


class TulpaValidateFileRequest(BaseRequestModel):
    path: str = ""


class TulpaRunTerminalRequest(BaseRequestModel):
    command: str = ""
    working_dir: str = "tulpa_stuff"
    timeout_seconds: int = 90


class TulpaReadFileQuery(BaseRequestModel):
    path: str = ""
    max_chars: int = Field(default=12000, ge=1, le=200000)
