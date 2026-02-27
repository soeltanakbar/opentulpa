"""Application-layer orchestration for directive/time profile APIs."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import Any

from opentulpa.application.contracts import ApplicationResult


class ProfileOrchestratorResult(ApplicationResult[dict[str, object]]):
    """Normalized route-friendly result payload."""


class ProfileOrchestrator:
    """Owns profile endpoint business rules independent of FastAPI transport."""

    def __init__(
        self,
        *,
        get_profiles: Callable[[], Any],
        get_memory: Callable[[], Any],
    ) -> None:
        self._get_profiles = get_profiles
        self._get_memory = get_memory

    def get_directive(self, *, customer_id: str) -> ProfileOrchestratorResult:
        profiles = self._get_profiles()
        safe_customer_id = str(customer_id).strip()
        return ProfileOrchestratorResult(
            status_code=200,
            payload={
                "customer_id": safe_customer_id,
                "directive": profiles.get_directive(safe_customer_id),
            },
        )

    def set_directive(
        self,
        *,
        customer_id: str,
        directive: str,
        source: str,
    ) -> ProfileOrchestratorResult:
        profiles = self._get_profiles()
        safe_customer_id = str(customer_id).strip()
        safe_directive = str(directive).strip()
        safe_source = str(source or "agent")
        profiles.set_directive(safe_customer_id, safe_directive, source=safe_source)

        memory = self._get_memory()
        if memory is not None:
            with suppress(Exception):
                memory.add_text(
                    f"Directive updated for this user: {safe_directive}",
                    user_id=safe_customer_id,
                    metadata={"kind": "directive_profile", "source": safe_source},
                )

        return ProfileOrchestratorResult(
            status_code=200,
            payload={"ok": True, "customer_id": safe_customer_id},
        )

    def clear_directive(self, *, customer_id: str) -> ProfileOrchestratorResult:
        profiles = self._get_profiles()
        safe_customer_id = str(customer_id).strip()
        cleared = profiles.clear_directive(safe_customer_id, source="agent")

        memory = self._get_memory()
        if memory is not None:
            with suppress(Exception):
                memory.add_text(
                    "Directive profile cleared for this user. Previous directive no longer applies.",
                    user_id=safe_customer_id,
                    metadata={"kind": "directive_profile", "source": "agent"},
                )

        return ProfileOrchestratorResult(
            status_code=200,
            payload={"ok": True, "customer_id": safe_customer_id, "cleared": cleared},
        )

    def get_time_profile(self, *, customer_id: str) -> ProfileOrchestratorResult:
        profiles = self._get_profiles()
        safe_customer_id = str(customer_id).strip()
        return ProfileOrchestratorResult(
            status_code=200,
            payload={
                "customer_id": safe_customer_id,
                "utc_offset": profiles.get_utc_offset(safe_customer_id),
            },
        )

    def set_time_profile(
        self,
        *,
        customer_id: str,
        utc_offset: str,
        source: str,
    ) -> ProfileOrchestratorResult:
        profiles = self._get_profiles()
        safe_customer_id = str(customer_id).strip()
        safe_utc_offset = str(utc_offset).strip()
        safe_source = str(source or "agent")
        normalized = profiles.set_utc_offset(safe_customer_id, safe_utc_offset, source=safe_source)
        return ProfileOrchestratorResult(
            status_code=200,
            payload={"ok": True, "customer_id": safe_customer_id, "utc_offset": normalized},
        )
