"""Thread-context compaction and rollup persistence helpers."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from opentulpa.agent.lc_messages import HumanMessage, SystemMessage
from opentulpa.agent.utils import (
    approx_tokens as _approx_tokens,
)
from opentulpa.agent.utils import (
    content_to_text as _content_to_text,
)
from opentulpa.agent.utils import (
    message_to_text as _message_to_text,
)


def split_text_chunks(text: str, *, approx_tokens_per_chunk: int = 25000) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    max_chars = max(12000, approx_tokens_per_chunk * 4)
    if len(raw) <= max_chars:
        return [raw]

    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0
    for para in raw.split("\n\n"):
        piece = para.strip()
        if not piece:
            continue
        piece_len = len(piece) + 2
        if current and current_chars + piece_len > max_chars:
            chunks.append("\n\n".join(current))
            current = [piece]
            current_chars = piece_len
        else:
            current.append(piece)
            current_chars += piece_len
    if current:
        chunks.append("\n\n".join(current))
    if not chunks:
        chunks = [raw[i : i + max_chars] for i in range(0, len(raw), max_chars)]
    return chunks


async def compress_rollup(runtime: Any, existing_rollup: str, additional_text: str) -> str:
    running = str(existing_rollup or "").strip()
    chunks = split_text_chunks(additional_text, approx_tokens_per_chunk=25000)
    if not chunks:
        return running
    for chunk in chunks:
        response = await runtime._model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You compress long-running assistant conversations into durable context.\n"
                        "Return plain text only. Preserve:\n"
                        "- user preferences/directives\n"
                        "- active goals and constraints\n"
                        "- important decisions and why\n"
                        "- unresolved tasks / follow-ups\n"
                        "- key facts with dates, IDs, links, and paths\n"
                        "Be concise and structured with short headings."
                    )
                ),
                HumanMessage(
                    content=(
                        "Existing compressed context (may be empty):\n"
                        f"{running[:12000]}\n\n"
                        "Older conversation segment to fold in:\n"
                        f"{chunk[:120000]}"
                    )
                ),
            ]
        )
        running = _content_to_text(getattr(response, "content", "")).strip() or running
    return running[:30000]


async def persist_rollup_memory(
    runtime: Any,
    *,
    customer_id: str,
    thread_id: str,
    rollup: str,
) -> None:
    cid = str(customer_id or "").strip()
    if not cid:
        return
    with suppress(Exception):
        await runtime._request_with_backoff(
            "POST",
            "/internal/memory/add",
            json_body={
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Thread context rollup updated for {thread_id}: "
                            f"{str(rollup or '')[:12000]}"
                        ),
                    }
                ],
                "user_id": cid,
                "metadata": {
                    "kind": "thread_context_rollup",
                    "thread_id": str(thread_id or ""),
                },
            },
            timeout=10.0,
            retries=1,
        )


async def maybe_compact_thread_context(
    runtime: Any,
    *,
    thread_id: str,
    customer_id: str,
) -> None:
    tid = str(thread_id or "").strip()
    if not tid:
        return
    if runtime._graph is None:
        return
    if runtime._checkpointer is None or not hasattr(runtime._checkpointer, "adelete_thread"):
        return

    config = {"configurable": {"thread_id": tid}, "recursion_limit": runtime.recursion_limit}
    for _ in range(3):
        try:
            snapshot = await runtime._graph.aget_state(config=config)
            values = getattr(snapshot, "values", {}) or {}
            state_messages = values.get("messages", [])
            if not isinstance(state_messages, list) or not state_messages:
                return
            message_texts = [_message_to_text(m) for m in state_messages]
            message_tokens = [_approx_tokens(t) for t in message_texts]
            total_tokens = sum(message_tokens)
            if total_tokens <= runtime._context_token_limit:
                return

            target_tokens = min(runtime._context_rollup_tokens, total_tokens)
            consumed = 0
            split_idx = 0
            for idx, tok in enumerate(message_tokens):
                consumed += tok
                split_idx = idx + 1
                if consumed >= target_tokens and split_idx < len(state_messages):
                    break
            if split_idx <= 0:
                return

            oldest_segment = "\n\n".join(message_texts[:split_idx]).strip()
            if not oldest_segment:
                return

            existing_rollup = runtime._load_thread_rollup(tid) or ""
            updated_rollup = await compress_rollup(runtime, existing_rollup, oldest_segment)
            if not updated_rollup:
                return

            runtime._save_thread_rollup(tid, updated_rollup)
            await persist_rollup_memory(
                runtime,
                customer_id=customer_id,
                thread_id=tid,
                rollup=updated_rollup,
            )

            remaining_messages = state_messages[split_idx:]
            await runtime._checkpointer.adelete_thread(tid)
            if remaining_messages:
                await runtime._graph.aupdate_state(
                    config=config,
                    values={"messages": remaining_messages},
                )
        except Exception:
            return
