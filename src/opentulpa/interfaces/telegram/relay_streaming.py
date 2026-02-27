"""Streaming relay helpers for Telegram."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from opentulpa.agent.runtime_input import MergedInputSuppressedError
from opentulpa.interfaces.telegram.client import TelegramClient

logger = logging.getLogger(__name__)


async def _emit_typing_until_done(
    *,
    client: TelegramClient,
    chat_id: int,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        with suppress(Exception):
            await client.send_chat_action(chat_id=chat_id, action="typing")
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=4.0)


async def stream_langgraph_reply_to_telegram(
    *,
    agent_runtime: Any,
    thread_id: str,
    customer_id: str,
    text: str,
    bot_token: str,
    chat_id: int,
    is_low_signal_reply: Callable[[str], bool],
    stream_wait_signal: str,
    stream_approval_handoff_signal: str,
    telegram_client_factory: Callable[[str], Any] | None = None,
    asyncio_module: Any | None = None,
) -> tuple[str | None, bool]:
    stream_message_id: int | None = None
    progress_message_id: int | None = None
    last_streamed = ""
    final_reply = None
    delivered_any = False
    progress_notified = False
    client_factory = telegram_client_factory or TelegramClient
    async_lib = asyncio_module or asyncio
    client = client_factory(bot_token)
    waiting_for_segment = True
    stream_state: dict[str, int | None] = {"message_id": stream_message_id}
    typing_stop = asyncio.Event()
    typing_task = asyncio.create_task(
        _emit_typing_until_done(client=client, chat_id=chat_id, stop_event=typing_stop)
    )
    suppressed = False
    first_token_timeout_s = 90.0
    first_token_retry_timeout_s = 180.0
    stream_idle_timeout_s = 180.0
    stream_idle_retry_timeout_s = 240.0
    consecutive_timeouts = 0
    max_consecutive_timeouts = 2
    next_chunk_task: asyncio.Task[Any] | None = None
    logger.info(
        "telegram.stream start chat_id=%s thread_id=%s customer_id=%s text_chars=%s",
        chat_id,
        thread_id,
        customer_id,
        len(str(text or "")),
    )

    async def _recover_after_stream_timeout() -> str | None:
        if not hasattr(agent_runtime, "ainvoke_text"):
            return None
        try:
            recovered = await async_lib.wait_for(
                agent_runtime.ainvoke_text(
                    thread_id=thread_id,
                    customer_id=customer_id,
                    text=text,
                ),
                timeout=90.0,
            )
        except Exception:
            return None
        safe = str(recovered or "").strip()
        if not safe or is_low_signal_reply(safe):
            return None
        return safe

    try:
        stream = agent_runtime.astream_text(
            thread_id=thread_id,
            customer_id=customer_id,
            text=text,
        )
        stream_iter = stream.__aiter__()
        while True:
            if not last_streamed:
                timeout_s = (
                    first_token_timeout_s
                    if consecutive_timeouts == 0
                    else first_token_retry_timeout_s
                )
            else:
                timeout_s = (
                    stream_idle_timeout_s
                    if consecutive_timeouts == 0
                    else stream_idle_retry_timeout_s
                )
            try:
                if next_chunk_task is None:
                    next_chunk_task = async_lib.create_task(stream_iter.__anext__())
                partial = await async_lib.wait_for(
                    async_lib.shield(next_chunk_task),
                    timeout=timeout_s,
                )
                next_chunk_task = None
            except StopAsyncIteration:
                next_chunk_task = None
                break
            except asyncio.TimeoutError:
                consecutive_timeouts += 1
                if consecutive_timeouts < max_consecutive_timeouts:
                    if not progress_notified:
                        progress_text = "Still working on this — one sec."
                        progress_message_id = (
                            await client.upsert_stream_message(
                                chat_id=chat_id,
                                text=progress_text,
                                message_id=None,
                                parse_mode="HTML",
                            )
                        )
                        stream_message_id = progress_message_id or stream_state.get("message_id")
                        stream_state["message_id"] = stream_message_id
                        if progress_message_id is not None:
                            progress_notified = True
                    logger.warning(
                        "telegram.stream timeout_retry chat_id=%s thread_id=%s customer_id=%s stage=%s",
                        chat_id,
                        thread_id,
                        customer_id,
                        "first_token" if not last_streamed else "idle",
                    )
                    continue
                if next_chunk_task is not None and not next_chunk_task.done():
                    next_chunk_task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        async with asyncio.timeout(1.0):
                            await next_chunk_task
                next_chunk_task = None
                with suppress(Exception):
                    async with asyncio.timeout(1.0):
                        await stream.aclose()
                stream_message_id = stream_state.get("message_id")
                recovered_text = await _recover_after_stream_timeout()
                if recovered_text:
                    logger.warning(
                        "telegram.stream timeout_recovered chat_id=%s thread_id=%s customer_id=%s stage=%s",
                        chat_id,
                        thread_id,
                        customer_id,
                        "first_token" if not last_streamed else "idle",
                    )
                    if progress_message_id is not None:
                        with suppress(Exception):
                            await client.delete_message(chat_id=chat_id, message_id=progress_message_id)
                        progress_message_id = None
                        stream_message_id = None
                        stream_state["message_id"] = None
                    stream_message_id = (
                        await client.upsert_stream_message(
                            chat_id=chat_id,
                            text=recovered_text,
                            message_id=stream_message_id,
                            parse_mode="HTML",
                        )
                        or stream_message_id
                    )
                    stream_state["message_id"] = stream_message_id
                    if stream_message_id is not None:
                        delivered_any = True
                        final_reply = recovered_text
                    else:
                        final_reply = None
                    break
                timeout_text = (
                    "Still working, but the model response timed out. "
                    "Please retry in a moment."
                )
                logger.error(
                    "telegram.stream timeout_fail chat_id=%s thread_id=%s customer_id=%s stage=%s",
                    chat_id,
                    thread_id,
                    customer_id,
                    "first_token" if not last_streamed else "idle",
                )
                stream_message_id = (
                    await client.upsert_stream_message(
                        chat_id=chat_id,
                        text=timeout_text,
                        message_id=stream_message_id,
                        parse_mode="HTML",
                    )
                    or stream_message_id
                )
                stream_state["message_id"] = stream_message_id
                if stream_message_id is not None:
                    delivered_any = True
                    final_reply = timeout_text
                else:
                    final_reply = None
                break
            if isinstance(partial, str) and partial == stream_wait_signal:
                if not waiting_for_segment:
                    waiting_for_segment = True
                    last_streamed = ""
                    stream_state["message_id"] = None
                continue
            if isinstance(partial, str) and partial == stream_approval_handoff_signal:
                suppressed = True
                final_reply = None
                break
            if not isinstance(partial, str):
                continue
            consecutive_timeouts = 0
            current = partial.strip()
            if not current or is_low_signal_reply(current) or current == last_streamed:
                continue
            if progress_message_id is not None:
                with suppress(Exception):
                    await client.delete_message(chat_id=chat_id, message_id=progress_message_id)
                if stream_state.get("message_id") == progress_message_id:
                    stream_state["message_id"] = None
                progress_message_id = None
            if last_streamed and not current.startswith(last_streamed):
                waiting_for_segment = True
                last_streamed = ""
                stream_state["message_id"] = None
            if waiting_for_segment:
                waiting_for_segment = False
            stream_message_id = stream_state.get("message_id")
            stream_message_id = (
                await client.upsert_stream_message(
                    chat_id=chat_id,
                    text=current,
                    message_id=stream_message_id,
                    parse_mode="HTML",
                )
                or stream_message_id
            )
            if stream_message_id is not None:
                delivered_any = True
            last_streamed = current
            final_reply = current
            stream_state["message_id"] = stream_message_id
    except MergedInputSuppressedError:
        logger.info(
            "telegram.stream suppressed_by_merge chat_id=%s thread_id=%s customer_id=%s",
            chat_id,
            thread_id,
            customer_id,
        )
        suppressed = True
    except Exception:
        if next_chunk_task is not None and not next_chunk_task.done():
            next_chunk_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                async with asyncio.timeout(1.0):
                    await next_chunk_task
        if not typing_stop.is_set():
            typing_stop.set()
        with suppress(Exception):
            await typing_task
        raise
    if next_chunk_task is not None and not next_chunk_task.done():
        next_chunk_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            async with asyncio.timeout(1.0):
                await next_chunk_task
    if not typing_stop.is_set():
        typing_stop.set()
    with suppress(Exception):
        await typing_task
    if not suppressed and final_reply and not delivered_any:
        final_reply = None
    if not suppressed and not final_reply:
        logger.error(
            "telegram.stream no_final_reply chat_id=%s thread_id=%s customer_id=%s",
            chat_id,
            thread_id,
            customer_id,
        )
        stream_message_id = stream_state.get("message_id")
        fallback_text = (
            "I couldn't produce a visible user-facing reply for that step "
            "(the model/tool loop ended without displayable output)."
        )
        stream_message_id = (
            await client.upsert_stream_message(
                chat_id=chat_id,
                text=fallback_text,
                message_id=stream_message_id,
                parse_mode="HTML",
            )
            or stream_message_id
        )
        stream_state["message_id"] = stream_message_id
        final_reply = fallback_text
    logger.info(
        "telegram.stream complete chat_id=%s thread_id=%s customer_id=%s suppressed=%s final_chars=%s",
        chat_id,
        thread_id,
        customer_id,
        suppressed,
        len(str(final_reply or "")),
    )
    return final_reply, suppressed
