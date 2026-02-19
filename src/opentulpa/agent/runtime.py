"""
In-process LangGraph runtime for OpenTulpa.

This replaces the Parlant subprocess/session model with a local StateGraph that:
- runs tool-calling in a bounded loop,
- persists thread state via SQLite checkpointer,
- supports token streaming for Telegram,
- and reuses existing /internal/* APIs as tool backends.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

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
from opentulpa.agent.graph_builder import build_runtime_graph
from opentulpa.agent.lc_messages import AIMessage, HumanMessage, SystemMessage
from opentulpa.agent.tools_registry import register_runtime_tools
from opentulpa.agent.utils import (
    content_to_text as _content_to_text,
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
from opentulpa.context.service import EventContextService
from opentulpa.context.thread_rollups import ThreadRollupService


class OpenTulpaLangGraphRuntime:
    def __init__(
        self,
        *,
        app_url: str,
        openrouter_api_key: str,
        model_name: str,
        checkpoint_db_path: str,
        recursion_limit: int = 30,
        context_events: EventContextService | None = None,
        customer_profile_service: CustomerProfileService | None = None,
        thread_rollup_service: ThreadRollupService | None = None,
        context_token_limit: int = 250000,
        context_rollup_tokens: int = 100000,
    ) -> None:
        self.app_url = app_url.rstrip("/")
        self.openrouter_api_key = openrouter_api_key
        self.model_name = _normalize_model_name(model_name)
        self.checkpoint_db_path = checkpoint_db_path
        self.recursion_limit = recursion_limit
        self._context_events = context_events
        self._customer_profile_service = customer_profile_service
        self._thread_rollup_service = thread_rollup_service
        self._context_token_limit = max(50000, int(context_token_limit))
        self._context_rollup_tokens = max(10000, int(context_rollup_tokens))

        self._model = init_chat_model(
            self.model_name,
            model_provider="openai",
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
        )

        self._checkpointer_cm: Any | None = None
        self._checkpointer: Any | None = None
        self._graph = None
        self._tools: dict[str, Any] = {}
        self._model_with_tools = None

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        raw = str(text or "").strip()
        if not raw:
            return None

    @staticmethod
    def _strip_internal_json_prefix(text: str) -> str:
        """
        Remove internal control JSON prefixes that can leak into streamed user output.

        Example internal payloads:
        - {"selected": []} from skill selector
        - {"notify_user": false, "reason": "..."} from wake classifier
        """
        raw = str(text or "")
        working = raw.lstrip()
        changed = False
        decoder = json.JSONDecoder()

        while working.startswith("{"):
            try:
                parsed, end_idx = decoder.raw_decode(working)
            except Exception:
                break

            is_internal_selector = (
                isinstance(parsed, dict)
                and set(parsed.keys()) == {"selected"}
                and isinstance(parsed.get("selected"), list)
            )
            is_internal_classifier = (
                isinstance(parsed, dict)
                and "notify_user" in parsed
                and set(parsed.keys()).issubset({"notify_user", "reason"})
            )
            if not (is_internal_selector or is_internal_classifier):
                break

            working = working[end_idx:].lstrip()
            changed = True

        return working if changed else raw
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(raw[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    @staticmethod
    def _format_pending_context(events: list[dict[str, Any]], *, payload_limit: int = 800) -> str:
        lines: list[str] = []
        for idx, event in enumerate(events, start=1):
            source = str(event.get("source", "event"))
            event_type = str(event.get("event_type", "update"))
            payload = event.get("payload", {})
            if isinstance(payload, dict):
                payload_text = json.dumps(payload, ensure_ascii=False)
            else:
                payload_text = str(payload)
            payload_text = " ".join(payload_text.split())
            if len(payload_text) > payload_limit:
                payload_text = payload_text[:payload_limit] + "..."
            lines.append(f"{idx}. [{source}/{event_type}] {payload_text}")
        return "\n".join(lines)

    def _prepend_pending_context(
        self,
        *,
        customer_id: str,
        text: str,
        include_pending_context: bool,
    ) -> tuple[str, int | None]:
        if not include_pending_context or self._context_events is None:
            return text, None
        pending = self._context_events.list_events(customer_id, limit=20)
        if not pending:
            return text, None
        through_id = int(pending[-1]["id"])
        wrapped = (
            "System context updates collected while the user was away:\n"
            f"{self._format_pending_context(pending)}\n\n"
            f"User message:\n{text}"
        )
        return wrapped, through_id

    async def _load_active_directive(self, customer_id: str) -> str | None:
        cid = str(customer_id or "").strip()
        if not cid:
            return None
        if self._customer_profile_service is not None:
            try:
                return self._customer_profile_service.get_directive(cid)
            except Exception:
                pass
        try:
            r = await self._request_with_backoff(
                "POST",
                "/internal/directive/get",
                json_body={"customer_id": cid},
                timeout=5.0,
                retries=1,
            )
            if r.status_code != 200:
                return None
            data = r.json()
            directive = str(data.get("directive") or "").strip()
            return directive or None
        except Exception:
            return None

    async def _load_user_utc_offset(self, customer_id: str) -> str | None:
        cid = str(customer_id or "").strip()
        if not cid:
            return None
        if self._customer_profile_service is not None:
            with suppress(Exception):
                return self._customer_profile_service.get_utc_offset(cid)
        try:
            r = await self._request_with_backoff(
                "POST",
                "/internal/time_profile/get",
                json_body={"customer_id": cid},
                timeout=5.0,
                retries=1,
            )
            if r.status_code != 200:
                return None
            data = r.json()
            offset = str(data.get("utc_offset") or "").strip()
            return offset or None
        except Exception:
            return None

    async def _list_available_skills(self, customer_id: str) -> list[dict[str, Any]]:
        cid = str(customer_id or "").strip()
        try:
            r = await self._request_with_backoff(
                "POST",
                "/internal/skills/list",
                json_body={
                    "customer_id": cid,
                    "include_global": True,
                    "include_disabled": False,
                    "limit": 200,
                },
                timeout=8.0,
                retries=1,
            )
            if r.status_code != 200:
                return []
            data = r.json()
            skills = data.get("skills", [])
            if not isinstance(skills, list):
                return []
            out: list[dict[str, Any]] = []
            for item in skills:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                desc = str(item.get("description", "")).strip()
                scope = str(item.get("scope", "")).strip() or "user"
                if not name or not desc:
                    continue
                out.append(
                    {
                        "name": name,
                        "description": desc,
                        "scope": scope,
                    }
                )
            return out
        except Exception:
            return []

    async def _select_relevant_skills(
        self,
        *,
        customer_id: str,
        query: str,
        candidates: list[dict[str, Any]],
        max_skills: int = 2,
    ) -> list[dict[str, Any]]:
        prompt_query = str(query or "").strip()
        if not prompt_query or not candidates:
            return []
        shortlist = candidates[:80]
        catalog = "\n".join(
            [
                f"- name={c['name']} scope={c['scope']} description={c['description'][:300]}"
                for c in shortlist
            ]
        )
        try:
            response = await self._model.ainvoke(
                [
                    SystemMessage(
                        content=(
                            "You select reusable skills for the current user request.\n"
                            "Return strict JSON object with key 'selected', an array of objects:\n"
                            "  {\"name\": string, \"score\": number, \"reason\": string}\n"
                            "Choose only skills that materially improve answer quality.\n"
                            "If none apply, return {\"selected\": []}."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"customer_id={customer_id}\n"
                            f"user_request={prompt_query[:2000]}\n\n"
                            f"available_skills:\n{catalog}"
                        )
                    ),
                ]
            )
            raw = _content_to_text(getattr(response, "content", "")).strip()
            parsed = self._extract_json_object(raw) or {}
            selected_raw = parsed.get("selected", [])
            if not isinstance(selected_raw, list):
                return []
            by_name = {c["name"]: c for c in shortlist}
            selected: list[dict[str, Any]] = []
            for item in selected_raw:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                if not name or name not in by_name:
                    continue
                score_raw = item.get("score", 0)
                try:
                    score = float(score_raw)
                except Exception:
                    score = 0.0
                if score < 0.45:
                    continue
                selected.append(
                    {
                        **by_name[name],
                        "score": score,
                        "reason": str(item.get("reason", "")).strip()[:300],
                    }
                )
            selected.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return selected[: max(1, min(int(max_skills), 3))]
        except Exception:
            return []

    async def _resolve_skill_context(self, customer_id: str, user_text: str) -> dict[str, Any]:
        cid = str(customer_id or "").strip()
        query = str(user_text or "").strip()
        if not cid or not query:
            return {"skill_names": [], "context": ""}
        candidates = await self._list_available_skills(cid)
        if not candidates:
            return {"skill_names": [], "context": ""}
        selected = await self._select_relevant_skills(
            customer_id=cid,
            query=query,
            candidates=candidates,
            max_skills=2,
        )
        if not selected:
            return {"skill_names": [], "context": ""}

        sections: list[str] = []
        skill_names: list[str] = []
        total_chars = 0
        max_total_chars = 36000
        for item in selected:
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            try:
                r = await self._request_with_backoff(
                    "POST",
                    "/internal/skills/get",
                    json_body={
                        "customer_id": cid,
                        "name": name,
                        "include_files": True,
                        "include_global": True,
                    },
                    timeout=8.0,
                    retries=1,
                )
                if r.status_code != 200:
                    continue
                payload = r.json()
                skill = payload.get("skill", {})
                if not isinstance(skill, dict):
                    continue
                skill_md = str(skill.get("skill_markdown", "")).strip()
                if not skill_md:
                    continue
                supporting = skill.get("supporting_files", {})
                snippet = (
                    f"Skill name: {name}\n"
                    f"Scope: {skill.get('scope', '')}\n"
                    f"Description: {skill.get('description', '')}\n"
                    f"Selection reason: {item.get('reason', '')}\n\n"
                    f"SKILL.md:\n{skill_md[:12000]}"
                )
                if isinstance(supporting, dict) and supporting:
                    snippet += "\n\nSupporting files (snippets):"
                    for rel_path, file_text in list(supporting.items())[:6]:
                        snippet += (
                            f"\n- {str(rel_path)}:\n"
                            f"{str(file_text or '')[:3000]}"
                        )
                if total_chars + len(snippet) > max_total_chars:
                    break
                sections.append(snippet)
                skill_names.append(name)
                total_chars += len(snippet)
            except Exception:
                continue
        context = "\n\n---\n\n".join(sections).strip()
        return {"skill_names": skill_names, "context": context}

    async def _build_live_time_context(self, customer_id: str) -> dict[str, str]:
        now_server = datetime.now().astimezone()
        now_utc = datetime.now(timezone.utc)
        server_offset = now_server.utcoffset() or timedelta()
        server_offset_minutes = int(server_offset.total_seconds() // 60)
        server_offset_text = _minutes_to_utc_offset(server_offset_minutes)

        user_offset_text = await self._load_user_utc_offset(customer_id)
        source = "profile"
        user_offset_minutes = (
            _utc_offset_to_minutes(user_offset_text) if user_offset_text else None
        )
        if user_offset_minutes is None:
            user_offset_minutes = server_offset_minutes
            user_offset_text = server_offset_text
            source = "fallback_server_timezone"

        user_local = now_utc + timedelta(minutes=user_offset_minutes)
        return {
            "server_time_local_iso": now_server.isoformat(),
            "server_time_utc_iso": now_utc.isoformat(),
            "server_utc_offset": server_offset_text,
            "user_time_local_iso": user_local.isoformat(),
            "user_utc_offset": user_offset_text,
            "user_time_source": source,
        }

    def _load_thread_rollup(self, thread_id: str) -> str | None:
        tid = str(thread_id or "").strip()
        if not tid or self._thread_rollup_service is None:
            return None
        try:
            return self._thread_rollup_service.get_rollup(tid)
        except Exception:
            return None

    def _save_thread_rollup(self, thread_id: str, rollup: str) -> None:
        tid = str(thread_id or "").strip()
        text = str(rollup or "").strip()
        if not tid or not text or self._thread_rollup_service is None:
            return
        with suppress(Exception):
            self._thread_rollup_service.set_rollup(tid, text)

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

    async def start(self) -> None:
        if self._graph is not None:
            return
        db_path = Path(self.checkpoint_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpointer_cm = AsyncSqliteSaver.from_conn_string(str(db_path))
        self._checkpointer = await self._checkpointer_cm.__aenter__()
        if hasattr(self._checkpointer, "setup"):
            maybe_coro = self._checkpointer.setup()
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro
        self._register_tools()
        self._model_with_tools = self._model.bind_tools(list(self._tools.values()))
        self._graph = self._build_graph()

    async def shutdown(self) -> None:
        if self._checkpointer_cm is not None:
            await self._checkpointer_cm.__aexit__(None, None, None)
        self._checkpointer_cm = None
        self._checkpointer = None
        self._graph = None

    def healthy(self) -> bool:
        return self._graph is not None

    async def ainvoke_text(
        self,
        *,
        thread_id: str,
        customer_id: str,
        text: str,
        include_pending_context: bool = True,
    ) -> str:
        await self.start()
        assert self._graph is not None
        await self._maybe_compact_thread_context(thread_id=thread_id, customer_id=customer_id)
        merged_text, through_id = self._prepend_pending_context(
            customer_id=customer_id,
            text=text,
            include_pending_context=include_pending_context,
        )
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": self.recursion_limit}
        result = await self._graph.ainvoke(
            {
                "messages": [HumanMessage(content=merged_text)],
                "customer_id": customer_id,
                "thread_id": thread_id,
                "tool_error_count": 0,
            },
            config=config,
        )
        messages = result.get("messages", [])
        for message in reversed(messages):
            if isinstance(message, AIMessage) and (message.content or "").strip():
                cleaned = self._strip_internal_json_prefix(str(message.content))
                if through_id is not None and self._context_events is not None:
                    self._context_events.clear_events(customer_id, through_id=through_id)
                return cleaned.strip()
        return "I ran into an issue and could not produce a final response yet."

    async def astream_text(
        self,
        *,
        thread_id: str,
        customer_id: str,
        text: str,
        include_pending_context: bool = True,
    ) -> AsyncIterator[str]:
        await self.start()
        assert self._graph is not None
        await self._maybe_compact_thread_context(thread_id=thread_id, customer_id=customer_id)
        merged_text, through_id = self._prepend_pending_context(
            customer_id=customer_id,
            text=text,
            include_pending_context=include_pending_context,
        )
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": self.recursion_limit}
        accumulated = ""
        async for message_chunk, metadata in self._graph.astream(
            {
                "messages": [HumanMessage(content=merged_text)],
                "customer_id": customer_id,
                "thread_id": thread_id,
                "tool_error_count": 0,
            },
            config=config,
            stream_mode="messages",
        ):
            if metadata.get("langgraph_node") != "agent":
                continue
            if message_chunk.content:
                accumulated += str(message_chunk.content)
                cleaned = self._strip_internal_json_prefix(accumulated)
                if cleaned.strip():
                    yield cleaned

        if accumulated and through_id is not None and self._context_events is not None:
            self._context_events.clear_events(customer_id, through_id=through_id)
        if not accumulated:
            final = await self.ainvoke_text(
                thread_id=thread_id,
                customer_id=customer_id,
                text=text,
                include_pending_context=include_pending_context,
            )
            yield final

    async def classify_wake_event(
        self,
        *,
        customer_id: str,
        event_label: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Let the model decide whether a wake event should interrupt the user now."""
        try:
            response = await self._model.ainvoke(
                [
                    SystemMessage(
                        content=(
                            "You classify background assistant events.\n"
                            "Return strict JSON with keys: notify_user (bool), reason (string).\n"
                            "Use notify_user=true only when immediate user attention is required."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"customer_id={customer_id}\n"
                            f"event_label={event_label}\n"
                            f"payload={json.dumps(payload, ensure_ascii=False)[:5000]}"
                        )
                    ),
                ]
            )
            raw = response.content if hasattr(response, "content") else str(response)
            raw_text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            parsed = self._extract_json_object(raw_text) or {}
            return {
                "notify_user": bool(parsed.get("notify_user", False)),
                "reason": str(parsed.get("reason", "")).strip()[:500],
            }
        except Exception as exc:
            return {"notify_user": False, "reason": f"classifier_error:{exc}"}

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
        retryable_status = {429, 500, 502, 503, 504}
        retryable_errors = (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
        )
        for attempt in range(retries + 1):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.request(
                        method=method,
                        url=f"{self.app_url}{path}",
                        params=params,
                        json=json_body,
                        timeout=timeout,
                    )
                if response.status_code in retryable_status and attempt < retries:
                    await asyncio.sleep(0.6 * (2**attempt))
                    continue
                return response
            except retryable_errors:
                if attempt < retries:
                    await asyncio.sleep(0.6 * (2**attempt))
                    continue
                raise
        raise RuntimeError("request retry loop exhausted")

    def _register_tools(self) -> None:
        self._tools = register_runtime_tools(self)

    def _build_graph(self):
        return build_runtime_graph(self)
