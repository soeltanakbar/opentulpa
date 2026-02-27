"""
In-process LangGraph runtime for OpenTulpa.

This replaces the Parlant subprocess/session model with a local StateGraph that:
- runs tool-calling in a bounded loop,
- persists thread state via SQLite checkpointer,
- supports token streaming for Telegram,
- and reuses existing /internal/* APIs as tool backends.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
from langchain.chat_models import init_chat_model

from opentulpa.agent.context_compaction import (
    compress_rollup as _compress_rollup,
)
from opentulpa.agent.context_compaction import (
    maybe_compact_thread_context as _maybe_compact_thread_context,
)
from opentulpa.agent.context_compaction import (
    persist_rollup_memory as _persist_rollup_memory,
)
from opentulpa.agent.context_compaction import (
    split_text_chunks as _split_text_chunks,
)
from opentulpa.agent.file_analysis import (
    analyze_uploaded_file as _analyze_uploaded_file,
)
from opentulpa.agent.file_analysis import (
    extract_docx_text as _extract_docx_text,
)
from opentulpa.agent.file_analysis import (
    extract_pdf_text as _extract_pdf_text,
)
from opentulpa.agent.file_analysis import (
    extract_uploaded_text as _extract_uploaded_text,
)
from opentulpa.agent.file_analysis import (
    summarize_uploaded_blob as _summarize_uploaded_blob,
)
from opentulpa.agent.file_analysis import (
    transcribe_audio_blob as _transcribe_audio_blob,
)
from opentulpa.agent.internal_api_client import InternalApiClient
from opentulpa.agent.result_models import (
    CompletionClaimVerification,
    GuardrailIntentDecision,
    ToolGuardrailDecision,
    WakeEventDecision,
)
from opentulpa.agent.runtime_behavior import (
    log_behavior_event as _log_behavior_event,
)
from opentulpa.agent.runtime_classification import (
    classify_guardrail_intent as _classify_guardrail_intent_facade,
)
from opentulpa.agent.runtime_classification import (
    classify_wake_event as _classify_wake_event_facade,
)
from opentulpa.agent.runtime_classification import (
    evaluate_tool_guardrail as _evaluate_tool_guardrail_facade,
)
from opentulpa.agent.runtime_classification import (
    verify_completion_claim as _verify_completion_claim_facade,
)
from opentulpa.agent.runtime_context_links import (
    build_link_alias_context as _build_link_alias_context,
)
from opentulpa.agent.runtime_context_links import (
    expand_link_aliases as _expand_link_aliases,
)
from opentulpa.agent.runtime_context_links import (
    prepend_pending_context as _prepend_pending_context,
)
from opentulpa.agent.runtime_context_links import (
    register_links_from_text as _register_links_from_text,
)
from opentulpa.agent.runtime_facade import (
    execute_tool as _execute_tool_facade,
)
from opentulpa.agent.runtime_facade import (
    request_with_backoff as _request_with_backoff_facade,
)
from opentulpa.agent.runtime_guardrails import (
    has_pending_approval_lock as _has_pending_approval_lock,
)
from opentulpa.agent.runtime_helpers import (
    extract_approval_handoff_payload as _extract_approval_handoff_payload,
)
from opentulpa.agent.runtime_helpers import (
    extract_json_object as _extract_json_object,
)
from opentulpa.agent.runtime_helpers import (
    format_approval_handoff_reply as _format_approval_handoff_reply,
)
from opentulpa.agent.runtime_helpers import (
    format_pending_context as _format_pending_context,
)
from opentulpa.agent.runtime_helpers import (
    has_incomplete_internal_json_prefix as _has_incomplete_internal_json_prefix,
)
from opentulpa.agent.runtime_helpers import (
    resolve_link_aliases_in_args as _resolve_link_aliases_in_args,
)
from opentulpa.agent.runtime_helpers import (
    strip_internal_json_prefix as _strip_internal_json_prefix,
)
from opentulpa.agent.runtime_input import (
    ThreadInputCoordinator,
)
from opentulpa.agent.runtime_lifecycle import (
    runtime_healthy as _runtime_healthy,
)
from opentulpa.agent.runtime_lifecycle import (
    shutdown_runtime as _shutdown_runtime,
)
from opentulpa.agent.runtime_lifecycle import (
    start_runtime as _start_runtime,
)
from opentulpa.agent.runtime_profile_skills import (
    list_available_skills as _list_available_skills,
)
from opentulpa.agent.runtime_profile_skills import (
    load_active_directive as _load_active_directive,
)
from opentulpa.agent.runtime_profile_skills import (
    load_user_utc_offset as _load_user_utc_offset,
)
from opentulpa.agent.runtime_profile_skills import (
    pre_resolve_skill_state as _pre_resolve_skill_state,
)
from opentulpa.agent.runtime_profile_skills import (
    resolve_skill_context as _resolve_skill_context,
)
from opentulpa.agent.runtime_profile_skills import (
    select_relevant_skills as _select_relevant_skills,
)
from opentulpa.agent.runtime_time_rollups import (
    build_live_time_context as _build_live_time_context,
)
from opentulpa.agent.runtime_time_rollups import (
    cap_rollup_text as _cap_rollup_text,
)
from opentulpa.agent.runtime_time_rollups import (
    load_thread_rollup as _load_thread_rollup,
)
from opentulpa.agent.runtime_time_rollups import (
    save_thread_rollup as _save_thread_rollup,
)
from opentulpa.agent.runtime_turns import (
    ainvoke_text as _ainvoke_text,
)
from opentulpa.agent.runtime_turns import (
    astream_text as _astream_text,
)
from opentulpa.agent.utils import (
    minutes_to_utc_offset as _minutes_to_utc_offset,
)
from opentulpa.agent.utils import (
    normalize_model_name as _normalize_model_name,
)
from opentulpa.agent.utils import (
    utc_offset_to_minutes as _utc_offset_to_minutes,
)
from opentulpa.context.customer_profiles import CustomerProfileService
from opentulpa.context.link_aliases import LinkAliasService
from opentulpa.context.service import EventContextService
from opentulpa.context.thread_rollups import ThreadRollupService

logger = logging.getLogger(__name__)
STREAM_WAIT_SIGNAL = "__TULPA_STREAM_WAIT__"
STREAM_APPROVAL_HANDOFF_SIGNAL = "__TULPA_APPROVAL_HANDOFF__"
STREAM_EMPTY_REPLY_FALLBACK = (
    "I couldn't produce a visible user-facing reply for that step. "
    "Please retry, and I will continue from the latest state."
)

APPROVAL_EXECUTION_CUSTOMER_ID_TOOLS: set[str] = {
    "memory_search",
    "memory_add",
    "uploaded_file_search",
    "uploaded_file_get",
    "uploaded_file_send",
    "tulpa_file_send",
    "web_image_send",
    "uploaded_file_analyze",
    "skill_list",
    "skill_get",
    "skill_upsert",
    "skill_delete",
    "directive_get",
    "directive_set",
    "directive_clear",
    "time_profile_get",
    "time_profile_set",
    "routine_list",
    "routine_create",
    "routine_delete",
    "automation_delete",
    "browser_use_run",
    "tulpa_run_terminal",
}


class OpenTulpaLangGraphRuntime:
    def __init__(
        self,
        *,
        app_url: str,
        openrouter_api_key: str,
        model_name: str,
        wake_classifier_model_name: str | None = None,
        guardrail_classifier_model_name: str | None = None,
        checkpoint_db_path: str,
        recursion_limit: int = 30,
        context_events: EventContextService | None = None,
        customer_profile_service: CustomerProfileService | None = None,
        thread_rollup_service: ThreadRollupService | None = None,
        link_alias_service: LinkAliasService | None = None,
        context_token_limit: int = 12000,
        context_rollup_tokens: int = 2200,
        context_recent_tokens: int = 3500,
        context_compaction_source_tokens: int = 100000,
        input_debounce_seconds: float = 0.65,
        proactive_heartbeat_default_hours: int = 3,
        behavior_log_enabled: bool = True,
        behavior_log_path: str = ".opentulpa/logs/agent_behavior.jsonl",
    ) -> None:
        self.app_url = app_url.rstrip("/")
        self.openrouter_api_key = openrouter_api_key
        self.model_name = _normalize_model_name(model_name)
        self._wake_classifier_model_name = (
            _normalize_model_name(wake_classifier_model_name)
            if str(wake_classifier_model_name or "").strip()
            else self.model_name
        )
        guardrail_model = (
            str(guardrail_classifier_model_name).strip()
            if str(guardrail_classifier_model_name or "").strip()
            else "minimax/minimax-m2.5"
        )
        self._guardrail_classifier_model_name = _normalize_model_name(guardrail_model)
        self.checkpoint_db_path = checkpoint_db_path
        self.recursion_limit = recursion_limit
        self._context_events = context_events
        self._customer_profile_service = customer_profile_service
        self._thread_rollup_service = thread_rollup_service
        self._link_alias_service = link_alias_service
        self._context_token_limit = max(6000, min(24000, int(context_token_limit)))
        self._context_short_term_high_tokens = self._context_token_limit
        self._context_short_term_low_tokens = min(
            max(1500, int(context_recent_tokens)),
            max(1500, self._context_short_term_high_tokens - 500),
        )
        self._context_rollup_tokens = min(
            max(500, int(context_rollup_tokens)),
            max(500, self._context_short_term_low_tokens - 250),
        )
        # Backward-compat aliases consumed by existing helpers/tests.
        self._context_recent_tokens = self._context_short_term_low_tokens
        self._context_compaction_source_tokens = max(
            self._context_rollup_tokens,
            int(context_compaction_source_tokens),
        )
        self._input_debounce_seconds = max(0.0, min(float(input_debounce_seconds), 3.0))
        self._proactive_heartbeat_default_hours = max(1, min(int(proactive_heartbeat_default_hours), 24))
        self._behavior_log_enabled = bool(behavior_log_enabled)
        raw_behavior_path = str(behavior_log_path or "").strip() or ".opentulpa/logs/agent_behavior.jsonl"
        self._behavior_log_path = Path(raw_behavior_path).resolve()
        self._behavior_log_lock = threading.Lock()
        if self._behavior_log_enabled:
            self._behavior_log_path.parent.mkdir(parents=True, exist_ok=True)

        self._model = init_chat_model(
            self.model_name,
            model_provider="openai",
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
        )
        if self._wake_classifier_model_name == self.model_name:
            self._wake_classifier_model = self._model
        else:
            try:
                self._wake_classifier_model = init_chat_model(
                    self._wake_classifier_model_name,
                    model_provider="openai",
                    api_key=openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1",
                    temperature=0,
                )
            except Exception:
                logger.exception(
                    "Failed to initialize wake classifier model '%s'; falling back to main model '%s'.",
                    self._wake_classifier_model_name,
                    self.model_name,
                )
                self._wake_classifier_model = self._model
        if self._guardrail_classifier_model_name == self.model_name:
            self._guardrail_classifier_model = self._model
        elif self._guardrail_classifier_model_name == self._wake_classifier_model_name:
            self._guardrail_classifier_model = self._wake_classifier_model
        else:
            try:
                self._guardrail_classifier_model = init_chat_model(
                    self._guardrail_classifier_model_name,
                    model_provider="openai",
                    api_key=openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1",
                    temperature=0,
                )
            except Exception:
                logger.exception(
                    "Failed to initialize guardrail classifier model '%s'; "
                    "falling back to main model '%s'.",
                    self._guardrail_classifier_model_name,
                    self.model_name,
                )
                self._guardrail_classifier_model = self._model

        self._checkpointer_cm: Any | None = None
        self._checkpointer: Any | None = None
        self._graph = None
        self._tools: dict[str, Any] = {}
        self._model_with_tools = None
        self._thread_inputs = ThreadInputCoordinator(debounce_seconds=self._input_debounce_seconds)
        self._internal_api = InternalApiClient(base_url=self.app_url)

    def log_behavior_event(self, *, event: str, **fields: Any) -> None:
        _log_behavior_event(
            behavior_log_enabled=bool(getattr(self, "_behavior_log_enabled", False)),
            event=event,
            fields=fields,
            behavior_log_path=getattr(self, "_behavior_log_path", None),
            behavior_log_lock=getattr(self, "_behavior_log_lock", None),
        )

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        return _extract_json_object(text)

    @staticmethod
    def _strip_internal_json_prefix(text: str) -> str:
        return _strip_internal_json_prefix(text)

    @staticmethod
    def _has_incomplete_internal_json_prefix(text: str) -> bool:
        return _has_incomplete_internal_json_prefix(text)

    @staticmethod
    def _format_pending_context(events: list[dict[str, Any]], *, payload_limit: int = 800) -> str:
        return _format_pending_context(events, payload_limit=payload_limit)

    def _prepend_pending_context(
        self,
        *,
        customer_id: str,
        text: str,
        include_pending_context: bool,
    ) -> tuple[str, int | None]:
        return _prepend_pending_context(
            context_events=self._context_events,
            customer_id=customer_id,
            text=text,
            include_pending_context=include_pending_context,
            format_pending_context=lambda events: self._format_pending_context(events),
        )

    async def _has_pending_approval_lock(self, *, customer_id: str, thread_id: str) -> bool:
        return await _has_pending_approval_lock(
            customer_id=customer_id,
            thread_id=thread_id,
            request_with_backoff=self._request_with_backoff,
        )

    @staticmethod
    def _extract_approval_handoff_payload(messages: list[Any]) -> dict[str, Any]:
        return _extract_approval_handoff_payload(messages)

    @staticmethod
    def _format_approval_handoff_reply(payload: dict[str, Any]) -> str:
        return _format_approval_handoff_reply(payload)

    def register_links_from_text(
        self,
        *,
        customer_id: str,
        text: str,
        source: str,
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        return _register_links_from_text(
            link_alias_service=self._link_alias_service,
            customer_id=customer_id,
            text=text,
            source=source,
            limit=limit,
        )

    def expand_link_aliases(self, *, customer_id: str, text: str) -> str:
        return _expand_link_aliases(
            link_alias_service=self._link_alias_service,
            customer_id=customer_id,
            text=text,
        )

    def resolve_link_aliases_in_args(self, *, customer_id: str, args: dict[str, Any]) -> dict[str, Any]:
        return _resolve_link_aliases_in_args(
            args=args,
            expand_alias_text=lambda value: self.expand_link_aliases(
                customer_id=customer_id,
                text=value,
            ),
        )

    def _build_link_alias_context(self, *, customer_id: str, user_text: str) -> str:
        return _build_link_alias_context(
            link_alias_service=self._link_alias_service,
            customer_id=customer_id,
            user_text=user_text,
        )

    async def _load_active_directive(self, customer_id: str) -> str | None:
        return await _load_active_directive(
            customer_id=customer_id,
            customer_profile_service=self._customer_profile_service,
            request_with_backoff=self._request_with_backoff,
        )

    async def _load_user_utc_offset(self, customer_id: str) -> str | None:
        return await _load_user_utc_offset(
            customer_id=customer_id,
            customer_profile_service=self._customer_profile_service,
            request_with_backoff=self._request_with_backoff,
        )

    async def _list_available_skills(self, customer_id: str) -> list[dict[str, Any]]:
        return await _list_available_skills(
            customer_id=customer_id,
            request_with_backoff=self._request_with_backoff,
        )

    async def _select_relevant_skills(
        self,
        *,
        customer_id: str,
        query: str,
        candidates: list[dict[str, Any]],
        max_skills: int = 2,
    ) -> list[dict[str, Any]]:
        return await _select_relevant_skills(
            customer_id=customer_id,
            query=query,
            candidates=candidates,
            model=self._model,
            extract_json_object=self._extract_json_object,
            max_skills=max_skills,
        )

    async def _resolve_skill_context(self, customer_id: str, user_text: str) -> dict[str, Any]:
        return await _resolve_skill_context(
            customer_id=customer_id,
            user_text=user_text,
            model=self._model,
            request_with_backoff=self._request_with_backoff,
            extract_json_object=self._extract_json_object,
        )

    async def _build_live_time_context(self, customer_id: str) -> dict[str, str]:
        return await _build_live_time_context(
            customer_id=customer_id,
            load_user_utc_offset=self._load_user_utc_offset,
            minutes_to_utc_offset=_minutes_to_utc_offset,
            utc_offset_to_minutes=_utc_offset_to_minutes,
        )

    def _load_thread_rollup(self, thread_id: str) -> str | None:
        return _load_thread_rollup(
            thread_id=thread_id,
            thread_rollup_service=self._thread_rollup_service,
            context_rollup_tokens=self._context_rollup_tokens,
        )

    def _save_thread_rollup(self, thread_id: str, rollup: str) -> None:
        _save_thread_rollup(
            thread_id=thread_id,
            rollup=rollup,
            thread_rollup_service=self._thread_rollup_service,
            context_rollup_tokens=self._context_rollup_tokens,
        )

    def _cap_rollup_text(self, text: str | None) -> str:
        return _cap_rollup_text(
            text=text,
            context_rollup_tokens=self._context_rollup_tokens,
        )

    @staticmethod
    def _extract_docx_text(raw_bytes: bytes) -> str:
        return _extract_docx_text(raw_bytes)

    @staticmethod
    def _extract_pdf_text(raw_bytes: bytes) -> str:
        return _extract_pdf_text(raw_bytes)

    @staticmethod
    def _extract_uploaded_text(
        *,
        raw_bytes: bytes,
        filename: str | None,
        mime_type: str | None,
        max_chars: int = 140000,
    ) -> str:
        return _extract_uploaded_text(
            raw_bytes=raw_bytes,
            filename=filename,
            mime_type=mime_type,
            max_chars=max_chars,
        )

    async def summarize_uploaded_blob(
        self,
        *,
        filename: str | None,
        mime_type: str | None,
        kind: str | None,
        raw_bytes: bytes,
        caption: str | None = None,
        question: str | None = None,
    ) -> str:
        return await _summarize_uploaded_blob(
            self,
            filename=filename,
            mime_type=mime_type,
            kind=kind,
            raw_bytes=raw_bytes,
            caption=caption,
            question=question,
        )

    async def transcribe_audio_blob(
        self,
        *,
        filename: str | None,
        mime_type: str | None,
        kind: str | None,
        raw_bytes: bytes,
    ) -> str:
        return await _transcribe_audio_blob(
            self,
            filename=filename,
            mime_type=mime_type,
            kind=kind,
            raw_bytes=raw_bytes,
        )

    async def analyze_uploaded_file(
        self,
        *,
        record: dict[str, Any],
        raw_bytes: bytes,
        question: str | None = None,
    ) -> dict[str, Any]:
        return await _analyze_uploaded_file(
            self,
            record=record,
            raw_bytes=raw_bytes,
            question=question,
        )

    @staticmethod
    def _split_text_chunks(text: str, *, approx_tokens_per_chunk: int = 25000) -> list[str]:
        return _split_text_chunks(text, approx_tokens_per_chunk=approx_tokens_per_chunk)

    async def _compress_rollup(self, existing_rollup: str, additional_text: str) -> str:
        return await _compress_rollup(self, existing_rollup, additional_text)

    async def _persist_rollup_memory(self, *, customer_id: str, thread_id: str, rollup: str) -> None:
        await _persist_rollup_memory(
            self,
            customer_id=customer_id,
            thread_id=thread_id,
            rollup=rollup,
        )

    async def _maybe_compact_thread_context(self, *, thread_id: str, customer_id: str) -> None:
        await _maybe_compact_thread_context(
            self,
            thread_id=thread_id,
            customer_id=customer_id,
        )

    async def _pre_resolve_skill_state(
        self,
        *,
        customer_id: str,
        user_text: str,
    ) -> dict[str, Any]:
        return await _pre_resolve_skill_state(
            customer_id=customer_id,
            user_text=user_text,
            model=self._model,
            request_with_backoff=self._request_with_backoff,
            extract_json_object=self._extract_json_object,
        )

    async def start(self) -> None:
        await _start_runtime(self)

    async def shutdown(self) -> None:
        await _shutdown_runtime(self)

    def healthy(self) -> bool:
        return _runtime_healthy(self)

    async def ainvoke_text(
        self,
        *,
        thread_id: str,
        customer_id: str,
        text: str,
        include_pending_context: bool = True,
        recursion_limit_override: int | None = None,
    ) -> str:
        return await _ainvoke_text(
            self,
            thread_id=thread_id,
            customer_id=customer_id,
            text=text,
            include_pending_context=include_pending_context,
            recursion_limit_override=recursion_limit_override,
        )

    async def astream_text(
        self,
        *,
        thread_id: str,
        customer_id: str,
        text: str,
        include_pending_context: bool = True,
    ) -> AsyncIterator[str]:
        async for chunk in _astream_text(
            self,
            thread_id=thread_id,
            customer_id=customer_id,
            text=text,
            stream_wait_signal=STREAM_WAIT_SIGNAL,
            stream_approval_handoff_signal=STREAM_APPROVAL_HANDOFF_SIGNAL,
            stream_empty_reply_fallback=STREAM_EMPTY_REPLY_FALLBACK,
            include_pending_context=include_pending_context,
        ):
            yield chunk

    async def classify_wake_event(
        self,
        *,
        customer_id: str,
        event_label: str,
        payload: dict[str, Any],
    ) -> WakeEventDecision:
        return await _classify_wake_event_facade(
            self,
            customer_id=customer_id,
            event_label=event_label,
            payload=payload,
        )

    async def verify_completion_claim(
        self,
        *,
        user_text: str,
        assistant_text: str,
        recent_tool_outputs: list[str],
        turn_window: str | None = None,
    ) -> CompletionClaimVerification:
        return await _verify_completion_claim_facade(
            self,
            user_text=user_text,
            assistant_text=assistant_text,
            recent_tool_outputs=recent_tool_outputs,
            turn_window=turn_window,
        )

    async def classify_guardrail_intent(
        self,
        *,
        action_name: str,
        action_args: dict[str, Any],
        action_note: str | None = None,
    ) -> GuardrailIntentDecision:
        return await _classify_guardrail_intent_facade(
            self,
            action_name=action_name,
            action_args=action_args,
            action_note=action_note,
        )

    async def evaluate_tool_guardrail(
        self,
        *,
        customer_id: str,
        thread_id: str,
        action_name: str,
        action_args: dict[str, Any],
        action_note: str | None = None,
    ) -> ToolGuardrailDecision:
        return await _evaluate_tool_guardrail_facade(
            self,
            customer_id=customer_id,
            thread_id=thread_id,
            action_name=action_name,
            action_args=action_args,
            action_note=action_note,
        )

    async def _request_with_backoff(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout: float = 20.0,
        retries: int = 2,
    ) -> httpx.Response:
        return await _request_with_backoff_facade(
            self,
            method,
            path,
            params=params,
            json_body=json_body,
            timeout=timeout,
            retries=retries,
        )

    async def execute_tool(
        self,
        *,
        action_name: str,
        action_args: dict[str, Any],
        customer_id: str | None = None,
        inject_customer_id: bool = False,
    ) -> Any:
        return await _execute_tool_facade(
            self,
            action_name=action_name,
            action_args=action_args,
            customer_id=customer_id,
            inject_customer_id=inject_customer_id,
            approval_execution_customer_id_tools=APPROVAL_EXECUTION_CUSTOMER_ID_TOOLS,
        )
