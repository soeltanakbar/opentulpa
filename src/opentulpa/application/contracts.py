"""Shared typed contracts for application-layer orchestrator results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

TPayload = TypeVar("TPayload")


@dataclass(slots=True)
class ApplicationResult(Generic[TPayload]):
    """Standardized orchestrator return contract."""

    status_code: int
    payload: TPayload
