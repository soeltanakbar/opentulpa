"""Turn-level invoke/stream orchestration helpers for runtime."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from opentulpa.agent.lc_messages import AIMessage, HumanMessage, ToolMessage
from opentulpa.agent.runtime_input import MergedInputSuppressedError
from opentulpa.agent.utils import content_to_text as _content_to_text
from opentulpa.core.ids import new_short_id

logger = logging.getLogger(__name__)


def _build_turn_graph_input(
    *,
    merged_text: str,
    customer_id: str,
    thread_id: str,
    turn_trace_id: str,
    skill_state: dict[str, Any],
    include_approval_handoff: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": [HumanMessage(content=merged_text)],
        "customer_id": customer_id,
        "thread_id": thread_id,
        "agent_trace_id": turn_trace_id,
        "tool_error_count": 0,
        "claim_check_retry_count": 0,
        "claim_check_needs_retry": False,
        **skill_state,
    }
    if include_approval_handoff:
        payload["approval_handoff"] = False
    return payload


async def ainvoke_text(
    runtime: Any,
    *,
    thread_id: str,
    customer_id: str,
    text: str,
    include_pending_context: bool = True,
    recursion_limit_override: int | None = None,
) -> str:
    await runtime.start()
    assert runtime._graph is not None
    turn_trace_id = new_short_id("turn")
    turn_state, effective_text = await runtime._thread_inputs.begin_turn(
        thread_id=thread_id,
        text=text,
    )
    if turn_state is None:
        runtime.log_behavior_event(
            event="turn_merged",
            trace_id=turn_trace_id,
            mode="ainvoke",
            thread_id=thread_id,
            customer_id=customer_id,
        )
        return ""
    try:
        runtime.log_behavior_event(
            event="turn_start",
            trace_id=turn_trace_id,
            mode="ainvoke",
            thread_id=thread_id,
            customer_id=customer_id,
            input_chars=len(str(effective_text or "")),
        )
        await runtime._maybe_compact_thread_context(thread_id=thread_id, customer_id=customer_id)
        merged_text, through_id = runtime._prepend_pending_context(
            customer_id=customer_id,
            text=effective_text,
            include_pending_context=include_pending_context,
        )
        if await runtime._has_pending_approval_lock(customer_id=customer_id, thread_id=thread_id):
            runtime.log_behavior_event(
                event="turn_blocked_pending_approval",
                trace_id=turn_trace_id,
                mode="ainvoke",
                thread_id=thread_id,
                customer_id=customer_id,
            )
            return ""
        runtime.register_links_from_text(
            customer_id=customer_id,
            text=merged_text,
            source="user_turn",
            limit=30,
        )
        merged_text = runtime.expand_link_aliases(customer_id=customer_id, text=merged_text)
        skill_state = await runtime._pre_resolve_skill_state(
            customer_id=customer_id,
            user_text=merged_text,
        )
        effective_recursion_limit = (
            max(5, min(int(recursion_limit_override), 200))
            if recursion_limit_override is not None
            else runtime.recursion_limit
        )
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": effective_recursion_limit,
        }
        result = await runtime._graph.ainvoke(
            _build_turn_graph_input(
                merged_text=merged_text,
                customer_id=customer_id,
                thread_id=thread_id,
                turn_trace_id=turn_trace_id,
                skill_state=skill_state,
                include_approval_handoff=True,
            ),
            config=config,
        )
        if bool(result.get("approval_handoff", False)):
            handoff_payload = runtime._extract_approval_handoff_payload(
                result.get("messages", [])
                if isinstance(result.get("messages", []), list)
                else []
            )
            handoff_text = runtime._format_approval_handoff_reply(handoff_payload)
            runtime.register_links_from_text(
                customer_id=customer_id,
                text=handoff_text,
                source="assistant_turn",
                limit=30,
            )
            runtime.log_behavior_event(
                event="turn_approval_handoff",
                trace_id=turn_trace_id,
                mode="ainvoke",
                thread_id=thread_id,
                customer_id=customer_id,
                output_chars=len(handoff_text.strip()),
            )
            return handoff_text
        messages = result.get("messages", [])
        for message in reversed(messages):
            if isinstance(message, AIMessage) and (message.content or "").strip():
                cleaned = runtime._strip_internal_json_prefix(str(message.content))
                if runtime._has_incomplete_internal_json_prefix(cleaned):
                    continue
                runtime.register_links_from_text(
                    customer_id=customer_id,
                    text=cleaned,
                    source="assistant_turn",
                    limit=30,
                )
                cleaned = runtime.expand_link_aliases(customer_id=customer_id, text=cleaned)
                if through_id is not None and runtime._context_events is not None:
                    runtime._context_events.clear_events(customer_id, through_id=through_id)
                runtime.log_behavior_event(
                    event="turn_complete",
                    trace_id=turn_trace_id,
                    mode="ainvoke",
                    thread_id=thread_id,
                    customer_id=customer_id,
                    output_chars=len(cleaned.strip()),
                )
                return cleaned.strip()
        runtime.log_behavior_event(
            event="turn_no_visible_reply",
            trace_id=turn_trace_id,
            mode="ainvoke",
            thread_id=thread_id,
            customer_id=customer_id,
        )
        return "I ran into an issue and could not produce a final response yet."
    except Exception as exc:
        runtime.log_behavior_event(
            event="turn_exception",
            trace_id=turn_trace_id,
            mode="ainvoke",
            thread_id=thread_id,
            customer_id=customer_id,
            error=str(exc)[:500],
        )
        raise
    finally:
        runtime._thread_inputs.end_turn(turn_state)


async def astream_text(
    runtime: Any,
    *,
    thread_id: str,
    customer_id: str,
    text: str,
    stream_wait_signal: str,
    stream_approval_handoff_signal: str,
    stream_empty_reply_fallback: str,
    include_pending_context: bool = True,
) -> AsyncIterator[str]:
    await runtime.start()
    assert runtime._graph is not None
    turn_trace_id = new_short_id("turn")
    turn_state, effective_text = await runtime._thread_inputs.begin_turn(
        thread_id=thread_id,
        text=text,
    )
    if turn_state is None:
        logger.info(
            "runtime.astream_text merged_input thread_id=%s customer_id=%s",
            thread_id,
            customer_id,
        )
        runtime.log_behavior_event(
            event="turn_merged",
            trace_id=turn_trace_id,
            mode="astream",
            thread_id=thread_id,
            customer_id=customer_id,
        )
        raise MergedInputSuppressedError("input merged into previous in-flight turn")
    try:
        logger.info(
            "runtime.astream_text start thread_id=%s customer_id=%s text_chars=%s",
            thread_id,
            customer_id,
            len(str(effective_text or "")),
        )
        runtime.log_behavior_event(
            event="turn_start",
            trace_id=turn_trace_id,
            mode="astream",
            thread_id=thread_id,
            customer_id=customer_id,
            input_chars=len(str(effective_text or "")),
        )
        await runtime._maybe_compact_thread_context(thread_id=thread_id, customer_id=customer_id)
        merged_text, through_id = runtime._prepend_pending_context(
            customer_id=customer_id,
            text=effective_text,
            include_pending_context=include_pending_context,
        )
        if await runtime._has_pending_approval_lock(customer_id=customer_id, thread_id=thread_id):
            yielded_any = True
            runtime.log_behavior_event(
                event="turn_blocked_pending_approval",
                trace_id=turn_trace_id,
                mode="astream",
                thread_id=thread_id,
                customer_id=customer_id,
            )
            yield stream_approval_handoff_signal
            return
        runtime.register_links_from_text(
            customer_id=customer_id,
            text=merged_text,
            source="user_turn",
            limit=30,
        )
        merged_text = runtime.expand_link_aliases(customer_id=customer_id, text=merged_text)
        skill_state = await runtime._pre_resolve_skill_state(
            customer_id=customer_id,
            user_text=merged_text,
        )
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": runtime.recursion_limit,
        }
        segment_accumulated = ""
        stream_key = ""
        yielded_any = False
        saw_agent_output = False
        in_tool_phase = False
        approval_handoff_detected = False

        def _finalize_segment() -> None:
            nonlocal segment_accumulated
            if not segment_accumulated:
                return
            cleaned_segment = runtime._strip_internal_json_prefix(segment_accumulated)
            if cleaned_segment.strip() and not runtime._has_incomplete_internal_json_prefix(
                cleaned_segment
            ):
                runtime.register_links_from_text(
                    customer_id=customer_id,
                    text=cleaned_segment,
                    source="assistant_turn",
                    limit=30,
                )
            segment_accumulated = ""

        async for message_chunk, metadata in runtime._graph.astream(
            _build_turn_graph_input(
                merged_text=merged_text,
                customer_id=customer_id,
                thread_id=thread_id,
                turn_trace_id=turn_trace_id,
                skill_state=skill_state,
                include_approval_handoff=False,
            ),
            config=config,
            stream_mode="messages",
        ):
            node_name = str(metadata.get("langgraph_node", "")).strip().lower()
            if node_name != "agent":
                if node_name == "tools":
                    tool_text = _content_to_text(getattr(message_chunk, "content", "")).strip()
                    if isinstance(message_chunk, ToolMessage) and tool_text.startswith(
                        "APPROVAL_HANDOFF"
                    ):
                        approval_handoff_detected = True
                if saw_agent_output and not in_tool_phase:
                    in_tool_phase = True
                    _finalize_segment()
                    yield stream_wait_signal
                continue
            if in_tool_phase:
                in_tool_phase = False
                stream_key = ""
                _finalize_segment()
            chunk_key = str(getattr(message_chunk, "id", "") or "")
            if chunk_key and stream_key and chunk_key != stream_key:
                _finalize_segment()
            if chunk_key:
                stream_key = chunk_key
            if message_chunk.content:
                saw_agent_output = True
                segment_accumulated += str(message_chunk.content)
                cleaned = runtime._strip_internal_json_prefix(segment_accumulated)
                if not cleaned.strip():
                    continue
                if cleaned == segment_accumulated and runtime._has_incomplete_internal_json_prefix(
                    segment_accumulated
                ):
                    continue
                expanded = runtime.expand_link_aliases(customer_id=customer_id, text=cleaned)
                if expanded.strip():
                    yielded_any = True
                    yield expanded

        if through_id is not None and runtime._context_events is not None:
            runtime._context_events.clear_events(customer_id, through_id=through_id)
        _finalize_segment()
        if not yielded_any and approval_handoff_detected:
            yielded_any = True
            runtime.log_behavior_event(
                event="turn_approval_handoff",
                trace_id=turn_trace_id,
                mode="astream",
                thread_id=thread_id,
                customer_id=customer_id,
            )
            yield stream_approval_handoff_signal
        if not yielded_any:
            logger.warning(
                "runtime.astream_text no_visible_chunks thread_id=%s customer_id=%s; invoking fallback",
                thread_id,
                customer_id,
            )
            runtime.log_behavior_event(
                event="turn_stream_no_visible_chunks",
                trace_id=turn_trace_id,
                thread_id=thread_id,
                customer_id=customer_id,
            )
            fallback_result = await runtime._graph.ainvoke(
                _build_turn_graph_input(
                    merged_text=merged_text,
                    customer_id=customer_id,
                    thread_id=thread_id,
                    turn_trace_id=turn_trace_id,
                    skill_state=skill_state,
                    include_approval_handoff=True,
                ),
                config=config,
            )
            if bool(fallback_result.get("approval_handoff", False)):
                yielded_any = True
                runtime.log_behavior_event(
                    event="turn_approval_handoff",
                    trace_id=turn_trace_id,
                    mode="astream",
                    thread_id=thread_id,
                    customer_id=customer_id,
                )
                yield stream_approval_handoff_signal
                fallback_result = {"messages": []}
            fallback_messages = fallback_result.get("messages", [])
            fallback_yielded = False
            for message in reversed(fallback_messages):
                if isinstance(message, AIMessage) and (message.content or "").strip():
                    cleaned = runtime._strip_internal_json_prefix(str(message.content))
                    if runtime._has_incomplete_internal_json_prefix(cleaned):
                        continue
                    if cleaned.strip():
                        runtime.register_links_from_text(
                            customer_id=customer_id,
                            text=cleaned,
                            source="assistant_turn",
                            limit=30,
                        )
                        cleaned = runtime.expand_link_aliases(
                            customer_id=customer_id,
                            text=cleaned,
                        )
                        fallback_yielded = True
                        runtime.log_behavior_event(
                            event="turn_stream_fallback_yielded",
                            trace_id=turn_trace_id,
                            thread_id=thread_id,
                            customer_id=customer_id,
                            output_chars=len(cleaned.strip()),
                        )
                        yield cleaned.strip()
                        break
            if not fallback_yielded:
                logger.error(
                    "runtime.astream_text fallback_no_ai_message thread_id=%s customer_id=%s messages_count=%s",
                    thread_id,
                    customer_id,
                    len(fallback_messages),
                )
                runtime.register_links_from_text(
                    customer_id=customer_id,
                    text=stream_empty_reply_fallback,
                    source="assistant_turn",
                    limit=5,
                )
                yielded_any = True
                runtime.log_behavior_event(
                    event="turn_stream_fallback_empty",
                    trace_id=turn_trace_id,
                    thread_id=thread_id,
                    customer_id=customer_id,
                )
                yield stream_empty_reply_fallback
        logger.info(
            "runtime.astream_text complete thread_id=%s customer_id=%s yielded_any=%s",
            thread_id,
            customer_id,
            yielded_any,
        )
        runtime.log_behavior_event(
            event="turn_complete",
            trace_id=turn_trace_id,
            mode="astream",
            thread_id=thread_id,
            customer_id=customer_id,
            yielded_any=yielded_any,
        )
    except Exception:
        logger.exception(
            "runtime.astream_text failed thread_id=%s customer_id=%s",
            thread_id,
            customer_id,
        )
        runtime.log_behavior_event(
            event="turn_exception",
            trace_id=turn_trace_id,
            mode="astream",
            thread_id=thread_id,
            customer_id=customer_id,
        )
        raise
    finally:
        runtime._thread_inputs.end_turn(turn_state)
