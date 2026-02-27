"""Application-layer orchestration for file-vault APIs."""

from __future__ import annotations

import mimetypes
from collections.abc import Callable
from typing import Any

from opentulpa.api.file_helpers import (
    download_image_from_web_url,
    sanitize_uploaded_file_record,
)
from opentulpa.application.contracts import ApplicationResult
from opentulpa.tasks.sandbox import TULPA_STUFF_DIR, is_within

MAX_LOCAL_SEND_BYTES = 45_000_000


class FileOrchestratorResult(ApplicationResult[dict[str, object]]):
    """Normalized route-friendly result payload."""


class FileOrchestrator:
    """Owns file-vault endpoint business rules independent of FastAPI transport."""

    def __init__(
        self,
        *,
        get_file_vault: Callable[[], Any],
        get_telegram_chat: Callable[[], Any],
        get_telegram_client: Callable[[], Any],
        get_agent_runtime: Callable[[], Any],
        telegram_enabled: bool,
    ) -> None:
        self._get_file_vault = get_file_vault
        self._get_telegram_chat = get_telegram_chat
        self._get_telegram_client = get_telegram_client
        self._get_agent_runtime = get_agent_runtime
        self._telegram_enabled = bool(telegram_enabled)

    def _resolve_chat_id(
        self,
        *,
        customer_id: str,
        fallback_record: dict[str, object] | None = None,
    ) -> Any:
        chat_id = None
        if isinstance(fallback_record, dict):
            chat_id = fallback_record.get("chat_id")
        if chat_id is None:
            slots = self._get_telegram_chat().find_session_slots(customer_id)
            if slots:
                chat_id = slots[0].get("chat_id")
        return chat_id

    def search_files(
        self,
        *,
        customer_id: str,
        query: str,
        limit: int,
    ) -> FileOrchestratorResult:
        vault = self._get_file_vault()
        results = [
            sanitize_uploaded_file_record(record, include_excerpt=False)
            for record in vault.search(customer_id, query=query, limit=limit)
        ]
        return FileOrchestratorResult(status_code=200, payload={"ok": True, "results": results})

    def get_file(
        self,
        *,
        customer_id: str,
        file_id: str,
        max_excerpt_chars: int,
    ) -> FileOrchestratorResult:
        vault = self._get_file_vault()
        record = vault.get_file(customer_id, file_id)
        if not record:
            return FileOrchestratorResult(status_code=404, payload={"detail": "file not found"})
        return FileOrchestratorResult(
            status_code=200,
            payload={
                "ok": True,
                "file": sanitize_uploaded_file_record(
                    record,
                    include_excerpt=True,
                    max_excerpt_chars=max_excerpt_chars,
                ),
            },
        )

    async def send_file(
        self,
        *,
        customer_id: str,
        file_id: str,
        caption: str | None,
    ) -> FileOrchestratorResult:
        if not self._telegram_enabled:
            return FileOrchestratorResult(status_code=501, payload={"detail": "Telegram not configured"})
        if not customer_id or not file_id:
            return FileOrchestratorResult(
                status_code=400,
                payload={"detail": "customer_id and file_id are required"},
            )

        vault = self._get_file_vault()
        record = vault.get_file(customer_id, file_id)
        if not record:
            return FileOrchestratorResult(status_code=404, payload={"detail": "file not found"})
        chat_id = self._resolve_chat_id(customer_id=customer_id, fallback_record=record)
        if chat_id is None:
            return FileOrchestratorResult(
                status_code=404,
                payload={"detail": "no chat found for customer"},
            )
        raw_bytes = vault.read_file_bytes(customer_id, file_id)
        if raw_bytes is None:
            return FileOrchestratorResult(
                status_code=404,
                payload={"detail": "stored file bytes not found"},
            )
        sent = await self._get_telegram_client().send_file(
            chat_id=chat_id,
            filename=str(record.get("original_filename", "file.bin")),
            raw_bytes=raw_bytes,
            kind=str(record.get("kind", "document")),
            mime_type=str(record.get("mime_type", "")).strip() or None,
            caption=caption,
            parse_mode="HTML",
        )
        if not sent:
            return FileOrchestratorResult(status_code=502, payload={"detail": "telegram send failed"})
        return FileOrchestratorResult(
            status_code=200,
            payload={"ok": True, "file_id": file_id, "chat_id": chat_id},
        )

    async def send_local_file(
        self,
        *,
        customer_id: str,
        local_path: str,
        caption: str | None,
    ) -> FileOrchestratorResult:
        if not self._telegram_enabled:
            return FileOrchestratorResult(status_code=501, payload={"detail": "Telegram not configured"})
        if not customer_id or not local_path:
            return FileOrchestratorResult(
                status_code=400,
                payload={"detail": "customer_id and path are required"},
            )
        try:
            target = (TULPA_STUFF_DIR.parent / local_path).resolve()
        except Exception:
            return FileOrchestratorResult(status_code=400, payload={"detail": "invalid path"})
        if not is_within(target, TULPA_STUFF_DIR):
            return FileOrchestratorResult(
                status_code=400,
                payload={"detail": "path must be under tulpa_stuff/"},
            )
        if not target.exists():
            return FileOrchestratorResult(status_code=404, payload={"detail": "file not found"})
        if target.is_dir():
            return FileOrchestratorResult(status_code=400, payload={"detail": "path is a directory"})
        try:
            file_size = int(target.stat().st_size)
        except Exception:
            return FileOrchestratorResult(
                status_code=502,
                payload={"detail": "failed to stat local file"},
            )
        if file_size > MAX_LOCAL_SEND_BYTES:
            return FileOrchestratorResult(
                status_code=413,
                payload={
                    "detail": f"file too large ({file_size} bytes > {MAX_LOCAL_SEND_BYTES} bytes)"
                },
            )

        chat_id = self._resolve_chat_id(customer_id=customer_id)
        if chat_id is None:
            return FileOrchestratorResult(
                status_code=404,
                payload={"detail": "no chat found for customer"},
            )
        try:
            raw_bytes = target.read_bytes()
        except Exception:
            return FileOrchestratorResult(
                status_code=502,
                payload={"detail": "failed to read local file"},
            )
        guessed_mime, _ = mimetypes.guess_type(str(target.name))
        sent = await self._get_telegram_client().send_file(
            chat_id=chat_id,
            filename=str(target.name),
            raw_bytes=raw_bytes,
            kind="document",
            mime_type=str(guessed_mime).strip() if guessed_mime else None,
            caption=caption,
            parse_mode="HTML",
        )
        if not sent:
            return FileOrchestratorResult(status_code=502, payload={"detail": "telegram send failed"})
        return FileOrchestratorResult(
            status_code=200,
            payload={"ok": True, "path": local_path, "chat_id": chat_id},
        )

    async def send_web_image(
        self,
        *,
        customer_id: str,
        image_url: str,
        caption: str | None,
        max_bytes: int,
    ) -> FileOrchestratorResult:
        if not self._telegram_enabled:
            return FileOrchestratorResult(status_code=501, payload={"detail": "Telegram not configured"})
        if not customer_id or not image_url:
            return FileOrchestratorResult(
                status_code=400,
                payload={"detail": "customer_id and url are required"},
            )
        chat_id = self._resolve_chat_id(customer_id=customer_id)
        if chat_id is None:
            return FileOrchestratorResult(
                status_code=404,
                payload={"detail": "no chat found for customer"},
            )
        try:
            downloaded = await download_image_from_web_url(image_url, max_bytes=max_bytes)
        except ValueError as exc:
            return FileOrchestratorResult(status_code=400, payload={"detail": str(exc)})
        except Exception as exc:
            return FileOrchestratorResult(
                status_code=502,
                payload={"detail": f"image fetch failed: {exc}"},
            )
        sent = await self._get_telegram_client().send_file(
            chat_id=chat_id,
            filename=str(downloaded["filename"]),
            raw_bytes=downloaded["raw_bytes"],
            kind=(
                "animation"
                if str(downloaded["content_type"]).strip().lower() == "image/gif"
                else "photo"
            ),
            mime_type=str(downloaded["content_type"]),
            caption=caption,
            parse_mode="HTML",
        )
        if not sent:
            return FileOrchestratorResult(status_code=502, payload={"detail": "telegram send failed"})
        return FileOrchestratorResult(
            status_code=200,
            payload={
                "ok": True,
                "chat_id": chat_id,
                "url": str(downloaded["final_url"]),
                "mime_type": str(downloaded["content_type"]),
                "size_bytes": int(downloaded["size_bytes"]),
            },
        )

    async def analyze_file(
        self,
        *,
        customer_id: str,
        file_id: str,
        question: str | None,
    ) -> FileOrchestratorResult:
        if not customer_id or not file_id:
            return FileOrchestratorResult(
                status_code=400,
                payload={"detail": "customer_id and file_id are required"},
            )
        agent_runtime = self._get_agent_runtime()
        if agent_runtime is None or not hasattr(agent_runtime, "analyze_uploaded_file"):
            return FileOrchestratorResult(
                status_code=501,
                payload={"detail": "agent runtime unavailable for file analysis"},
            )
        vault = self._get_file_vault()
        record = vault.get_file(customer_id, file_id)
        if not record:
            return FileOrchestratorResult(status_code=404, payload={"detail": "file not found"})
        raw_bytes = vault.read_file_bytes(customer_id, file_id)
        if raw_bytes is None:
            return FileOrchestratorResult(
                status_code=404,
                payload={"detail": "stored file bytes not found"},
            )
        try:
            analysis_result = await agent_runtime.analyze_uploaded_file(
                record=record,
                raw_bytes=raw_bytes,
                question=question,
            )
        except Exception as exc:
            return FileOrchestratorResult(
                status_code=500,
                payload={"detail": f"file analysis failed: {exc}"},
            )
        if not question:
            analysis_text = str(analysis_result.get("analysis", "")).strip()
            if analysis_text:
                updated = vault.set_ai_summary(customer_id, file_id, analysis_text)
                if isinstance(updated, dict):
                    record = updated
        return FileOrchestratorResult(
            status_code=200,
            payload={
                "ok": True,
                "analysis": str(analysis_result.get("analysis", "")).strip(),
                "file": sanitize_uploaded_file_record(
                    record,
                    include_excerpt=True,
                    max_excerpt_chars=16000,
                ),
            },
        )
