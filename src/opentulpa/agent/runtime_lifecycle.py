"""Runtime lifecycle orchestration helpers."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from opentulpa.agent.graph_builder import build_runtime_graph
from opentulpa.agent.tools_registry import register_runtime_tools


async def start_runtime(runtime: Any) -> None:
    if runtime._graph is not None:
        return
    db_path = Path(runtime.checkpoint_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    runtime._checkpointer_cm = AsyncSqliteSaver.from_conn_string(str(db_path))
    runtime._checkpointer = await runtime._checkpointer_cm.__aenter__()
    if hasattr(runtime._checkpointer, "setup"):
        maybe_coro = runtime._checkpointer.setup()
        if asyncio.iscoroutine(maybe_coro):
            await maybe_coro
    runtime._tools = register_runtime_tools(runtime)
    runtime._model_with_tools = runtime._model.bind_tools(list(runtime._tools.values()))
    runtime._graph = build_runtime_graph(runtime)


async def shutdown_runtime(runtime: Any) -> None:
    if runtime._checkpointer_cm is not None:
        await runtime._checkpointer_cm.__aexit__(None, None, None)
    runtime._checkpointer_cm = None
    runtime._checkpointer = None
    runtime._graph = None


def runtime_healthy(runtime: Any) -> bool:
    return runtime._graph is not None
