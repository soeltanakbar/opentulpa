"""Directive and time-profile route registration."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import Any

from fastapi import FastAPI, Request


def register_profile_routes(
    app: FastAPI,
    *,
    get_profiles: Callable[[], Any],
    get_memory: Callable[[], Any],
) -> None:
    """Register directive + timezone profile endpoints."""

    @app.post("/internal/directive/get")
    async def internal_directive_get(request: Request) -> Any:
        profiles = get_profiles()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        return {
            "customer_id": customer_id,
            "directive": profiles.get_directive(customer_id),
        }

    @app.post("/internal/directive/set")
    async def internal_directive_set(request: Request) -> Any:
        profiles = get_profiles()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        directive = str(body.get("directive", "")).strip()
        source = str(body.get("source", "agent") or "agent")
        profiles.set_directive(customer_id, directive, source=source)

        # Best-effort memory signal for recall; directive DB remains source of truth.
        memory = get_memory()
        if memory is not None:
            with suppress(Exception):
                memory.add_text(
                    f"Directive updated for this user: {directive}",
                    user_id=customer_id,
                    metadata={"kind": "directive_profile", "source": source},
                )

        return {"ok": True, "customer_id": customer_id}

    @app.post("/internal/directive/clear")
    async def internal_directive_clear(request: Request) -> Any:
        profiles = get_profiles()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        cleared = profiles.clear_directive(customer_id, source="agent")

        # Best-effort memory signal for recall; directive DB remains source of truth.
        memory = get_memory()
        if memory is not None:
            with suppress(Exception):
                memory.add_text(
                    "Directive profile cleared for this user. Previous directive no longer applies.",
                    user_id=customer_id,
                    metadata={"kind": "directive_profile", "source": "agent"},
                )

        return {"ok": True, "customer_id": customer_id, "cleared": cleared}

    @app.post("/internal/time_profile/get")
    async def internal_time_profile_get(request: Request) -> Any:
        profiles = get_profiles()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        return {
            "customer_id": customer_id,
            "utc_offset": profiles.get_utc_offset(customer_id),
        }

    @app.post("/internal/time_profile/set")
    async def internal_time_profile_set(request: Request) -> Any:
        profiles = get_profiles()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        utc_offset = str(body.get("utc_offset", "")).strip()
        source = str(body.get("source", "agent") or "agent")
        normalized = profiles.set_utc_offset(customer_id, utc_offset, source=source)
        return {"ok": True, "customer_id": customer_id, "utc_offset": normalized}
