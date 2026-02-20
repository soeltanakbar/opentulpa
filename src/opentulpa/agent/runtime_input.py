"""Thread-level turn input coordination for runtime debounce and coalescing."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field


class MergedInputSuppressedError(Exception):
    """Raised when a queued input was already merged into a previous in-flight turn."""


@dataclass
class _ThreadInputState:
    turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_inputs: list[tuple[str, str]] = field(default_factory=list)


class ThreadInputCoordinator:
    """Per-thread input queue with start-time debounce + coalescing semantics."""

    def __init__(self, *, debounce_seconds: float = 0.65) -> None:
        self._debounce_seconds = max(0.0, min(float(debounce_seconds), 3.0))
        self._states_lock = asyncio.Lock()
        self._states: dict[str, _ThreadInputState] = {}

    async def _get_state(self, thread_id: str) -> _ThreadInputState:
        tid = str(thread_id or "").strip() or "__default__"
        async with self._states_lock:
            state = self._states.get(tid)
            if state is None:
                state = _ThreadInputState()
                self._states[tid] = state
            return state

    async def begin_turn(self, *, thread_id: str, text: str) -> tuple[_ThreadInputState | None, str]:
        """
        Returns `(state, merged_text)`.

        If state is `None`, this request was already merged into an earlier in-flight turn.
        """
        state = await self._get_state(thread_id)
        request_id = f"req_{id(asyncio.current_task())}"
        safe_text = str(text or "").strip()
        async with state.pending_lock:
            state.pending_inputs.append((request_id, safe_text))

        await state.turn_lock.acquire()
        try:
            if self._debounce_seconds > 0:
                await asyncio.sleep(self._debounce_seconds)
            async with state.pending_lock:
                ids = [rid for rid, _ in state.pending_inputs]
                if request_id not in ids:
                    state.turn_lock.release()
                    return None, ""
                batch = state.pending_inputs[:]
                state.pending_inputs.clear()
            parts = [chunk.strip() for _, chunk in batch if chunk.strip()]
            merged_text = "\n\n".join(parts).strip()
            if not merged_text:
                merged_text = safe_text
            return state, merged_text
        except Exception:
            with suppress(Exception):
                state.turn_lock.release()
            raise

    @staticmethod
    def end_turn(state: _ThreadInputState | None) -> None:
        if state is None:
            return
        with suppress(Exception):
            state.turn_lock.release()
