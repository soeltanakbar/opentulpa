"""Application-layer orchestration for memory APIs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opentulpa.application.contracts import ApplicationResult


class MemoryOrchestratorResult(ApplicationResult[dict[str, object]]):
    """Normalized route-friendly result payload."""


class MemoryOrchestrator:
    """Owns memory endpoint business rules independent of FastAPI transport."""

    def __init__(self, *, get_memory: Callable[[], Any]) -> None:
        self._get_memory = get_memory

    def add_memory(
        self,
        *,
        messages: list[dict[str, object]],
        user_id: str | None,
        metadata: dict[str, object],
        infer: bool,
        retries: int,
    ) -> MemoryOrchestratorResult:
        mem = self._get_memory()
        resolved_user_id = user_id or mem.user_id
        result = mem.add(
            messages,
            user_id=resolved_user_id,
            metadata=metadata,
            infer=bool(infer),
            retries=int(retries or 1),
        )
        return MemoryOrchestratorResult(status_code=200, payload={"ok": True, "result": result})

    def search_memory(
        self,
        *,
        query: str,
        user_id: str | None,
        limit: int,
    ) -> MemoryOrchestratorResult:
        mem = self._get_memory()
        resolved_user_id = user_id or mem.user_id
        results = mem.search(query, user_id=resolved_user_id, limit=limit)
        return MemoryOrchestratorResult(status_code=200, payload={"results": results})
