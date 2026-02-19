"""Health route registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI


def register_health_routes(
    app: FastAPI,
    *,
    get_agent_runtime: Callable[[], Any],
) -> None:
    """Register liveness and runtime-health endpoints."""

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/agent/healthz")
    async def agent_health() -> dict[str, Any]:
        runtime = get_agent_runtime()
        healthy = bool(runtime and getattr(runtime, "healthy", lambda: False)())
        return {"status": "ok" if healthy else "degraded", "backend": "langgraph"}
