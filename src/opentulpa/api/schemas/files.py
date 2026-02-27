"""Schemas for file routes."""

from __future__ import annotations

from opentulpa.api.schemas.common import BaseRequestModel


class FileSearchRequest(BaseRequestModel):
    customer_id: str = ""
    query: str = ""
    limit: int = 5


class FileGetRequest(BaseRequestModel):
    customer_id: str = ""
    file_id: str = ""
    max_excerpt_chars: int = 16000


class FileSendRequest(BaseRequestModel):
    customer_id: str = ""
    file_id: str = ""
    caption: str | None = None


class FileSendLocalRequest(BaseRequestModel):
    customer_id: str = ""
    path: str = ""
    caption: str | None = None


class FileSendWebImageRequest(BaseRequestModel):
    customer_id: str = ""
    url: str = ""
    caption: str | None = None
    max_bytes: int = 10_000_000


class FileAnalyzeRequest(BaseRequestModel):
    customer_id: str = ""
    file_id: str = ""
    question: str | None = None
