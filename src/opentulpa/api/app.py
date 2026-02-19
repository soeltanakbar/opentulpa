"""FastAPI application: health, internal API, Telegram webhook, and agent runtime."""

import logging
import mimetypes
import re
from contextlib import asynccontextmanager, suppress
from typing import Any
from urllib.parse import unquote, urlparse

import httpx
from fastapi import APIRouter, BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import JSONResponse

from opentulpa.api.tulpa_loader import TulpaRouterLoader
from opentulpa.context.customer_profiles import CustomerProfileService
from opentulpa.context.file_vault import FileVaultService
from opentulpa.context.service import EventContextService
from opentulpa.core.config import get_settings
from opentulpa.core.ids import new_short_id
from opentulpa.integrations.slack_client import (
    grant_slack_write_consent,
    has_slack_write_consent,
)
from opentulpa.integrations.web_search import web_search as run_web_search
from opentulpa.interfaces.telegram.chat_service import TelegramChatService
from opentulpa.interfaces.telegram.client import TelegramClient
from opentulpa.memory.service import MemoryService
from opentulpa.scheduler.service import SchedulerService
from opentulpa.skills.service import SkillStoreService
from opentulpa.tasks.sandbox import (
    ALLOWED_TERMINAL_COMMANDS,
    ALLOWED_TERMINAL_DIRS,
    PROJECT_ROOT,
    get_tulpa_catalog,
)
from opentulpa.tasks.sandbox import (
    delete_file as sandbox_delete_file,
)
from opentulpa.tasks.sandbox import (
    read_file as sandbox_read_file,
)
from opentulpa.tasks.sandbox import (
    run_terminal as sandbox_run_terminal,
)
from opentulpa.tasks.sandbox import (
    validate_generated_file as sandbox_validate_generated_file,
)
from opentulpa.tasks.sandbox import (
    write_file as sandbox_write_file,
)
from opentulpa.tasks.service import TaskService
from opentulpa.tasks.wake_queue import WakeQueueService

logger = logging.getLogger(__name__)

# Global refs for internal routes (injected by create_app)
_memory: MemoryService | None = None
_scheduler: SchedulerService | None = None
_slack: Any | None = None  # SlackClient from integrations.slack_client
_tasks: TaskService | None = None
_wake_queue: WakeQueueService | None = None
_tulpa_loader: TulpaRouterLoader | None = None
_agent_runtime: Any | None = None
_context_events: EventContextService | None = None
_profiles: CustomerProfileService | None = None
_file_vault: FileVaultService | None = None
_skill_store: SkillStoreService | None = None
_telegram_chat: TelegramChatService | None = None
_telegram_client: TelegramClient | None = None
_tulpa_router = APIRouter()


def _get_memory() -> MemoryService:
    if _memory is None:
        raise RuntimeError("MemoryService not initialized")
    return _memory


def _get_scheduler() -> SchedulerService:
    if _scheduler is None:
        raise RuntimeError("SchedulerService not initialized")
    return _scheduler


def _get_slack() -> Any:
    if _slack is None:
        raise RuntimeError("Slack not initialized (no SLACK_BOT_TOKEN)")
    return _slack


def _get_tulpa_loader() -> TulpaRouterLoader:
    if _tulpa_loader is None:
        raise RuntimeError("Tulpa loader not initialized")
    return _tulpa_loader


def _get_tasks() -> TaskService:
    if _tasks is None:
        raise RuntimeError("TaskService not initialized")
    return _tasks


def _get_wake_queue() -> WakeQueueService:
    if _wake_queue is None:
        raise RuntimeError("WakeQueueService not initialized")
    return _wake_queue


def _get_context_events() -> EventContextService:
    if _context_events is None:
        raise RuntimeError("EventContextService not initialized")
    return _context_events


def _get_profiles() -> CustomerProfileService:
    if _profiles is None:
        raise RuntimeError("CustomerProfileService not initialized")
    return _profiles


def _get_file_vault() -> FileVaultService:
    if _file_vault is None:
        raise RuntimeError("FileVaultService not initialized")
    return _file_vault


def _get_skill_store() -> SkillStoreService:
    if _skill_store is None:
        raise RuntimeError("SkillStoreService not initialized")
    return _skill_store


def _get_telegram_chat() -> TelegramChatService:
    if _telegram_chat is None:
        raise RuntimeError("TelegramChatService not initialized")
    return _telegram_chat


def _get_telegram_client() -> TelegramClient:
    if _telegram_client is None:
        raise RuntimeError("TelegramClient not initialized")
    return _telegram_client


def _sanitize_uploaded_file_record(
    record: dict[str, Any],
    *,
    include_excerpt: bool = False,
    max_excerpt_chars: int = 16000,
) -> dict[str, Any]:
    clean = {
        "id": record.get("id"),
        "customer_id": record.get("customer_id"),
        "chat_id": record.get("chat_id"),
        "telegram_file_id": record.get("telegram_file_id"),
        "kind": record.get("kind"),
        "original_filename": record.get("original_filename"),
        "mime_type": record.get("mime_type"),
        "size_bytes": record.get("size_bytes"),
        "caption": record.get("caption"),
        "summary": record.get("summary"),
        "created_at": record.get("created_at"),
    }
    if include_excerpt:
        excerpt = str(record.get("text_excerpt", "") or "")
        clean["text_excerpt"] = excerpt[:max_excerpt_chars]
    return clean


def _normalize_cleanup_paths(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        path = str(item or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _collect_routine_cleanup_paths(payload: dict[str, Any]) -> list[str]:
    if not isinstance(payload, dict):
        return []
    candidates: list[str] = []
    list_keys = ("cleanup_paths", "script_paths", "file_paths")
    scalar_keys = ("cleanup_path", "script_path", "file_path")
    for key in list_keys:
        candidates.extend(_normalize_cleanup_paths(payload.get(key)))
    for key in scalar_keys:
        raw = str(payload.get(key, "")).strip()
        if raw:
            candidates.append(raw)
    seen: set[str] = set()
    out: list[str] = []
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _safe_telegram_filename(name: str, *, fallback: str = "image.jpg") -> str:
    raw = str(name or "").strip()
    if not raw:
        return fallback
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    return (safe or fallback)[:180]


def _infer_image_filename(image_url: str, content_type: str) -> str:
    parsed = urlparse(image_url)
    candidate = unquote(str(parsed.path or "").split("/")[-1]).strip()
    safe = _safe_telegram_filename(candidate, fallback="")
    if safe and "." in safe:
        return safe
    ext = mimetypes.guess_extension(str(content_type or "").strip().lower()) or ".jpg"
    return _safe_telegram_filename(f"image{ext}")


async def _download_image_from_web_url(
    image_url: str,
    *,
    max_bytes: int = 10_000_000,
) -> dict[str, Any]:
    raw_url = str(image_url or "").strip()
    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("url must start with http:// or https://")

    safe_limit = max(250_000, min(int(max_bytes), 25_000_000))
    timeout = httpx.Timeout(45.0, connect=10.0, read=45.0)
    headers = {"User-Agent": "OpenTulpa/0.1 (+send-web-image)"}
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
        try:
            head = await client.head(raw_url)
            if head.status_code < 400:
                head_type = str(head.headers.get("content-type", "")).split(";")[0].strip().lower()
                if head_type and not head_type.startswith("image/"):
                    raise ValueError(f"url does not point to an image (content-type={head_type})")
                head_len = str(head.headers.get("content-length", "")).strip()
                if head_len.isdigit() and int(head_len) > safe_limit:
                    raise ValueError(f"image too large ({head_len} bytes > {safe_limit} bytes)")
        except ValueError:
            raise
        except Exception:
            # Some origins reject HEAD; proceed with GET validation.
            pass

        async with client.stream("GET", raw_url) as resp:
            if resp.status_code >= 400:
                raise ValueError(f"image fetch failed: HTTP {resp.status_code}")
            ctype = str(resp.headers.get("content-type", "")).split(";")[0].strip().lower()
            if not ctype.startswith("image/"):
                raise ValueError(f"url does not point to an image (content-type={ctype or 'unknown'})")
            chunks: list[bytes] = []
            total = 0
            async for chunk in resp.aiter_bytes():
                if not chunk:
                    continue
                total += len(chunk)
                if total > safe_limit:
                    raise ValueError(f"image too large (>{safe_limit} bytes)")
                chunks.append(chunk)
            raw_bytes = b"".join(chunks)
            if not raw_bytes:
                raise ValueError("image fetch returned empty body")
            final_url = str(resp.url)

    filename = _infer_image_filename(final_url, ctype)
    return {
        "raw_bytes": raw_bytes,
        "content_type": ctype,
        "filename": filename,
        "final_url": final_url,
        "size_bytes": len(raw_bytes),
    }


def create_app(
    memory: MemoryService | None = None,
    scheduler: SchedulerService | None = None,
    slack_client: Any | None = None,
    task_service: TaskService | None = None,
    agent_runtime: Any | None = None,
    context_events: EventContextService | None = None,
    customer_profile_service: CustomerProfileService | None = None,
    file_vault_service: FileVaultService | None = None,
    skill_store_service: SkillStoreService | None = None,
) -> FastAPI:
    """Create FastAPI app with internal API, webhook, and agent runtime."""
    global _memory, _scheduler, _slack, _tulpa_loader, _tasks, _wake_queue, _agent_runtime, _context_events, _profiles, _file_vault, _skill_store, _telegram_chat, _telegram_client
    _memory = memory
    _scheduler = scheduler
    _slack = slack_client
    _tasks = task_service
    _agent_runtime = agent_runtime
    _context_events = context_events or EventContextService(
        db_path=PROJECT_ROOT / ".opentulpa" / "context_events.db"
    )
    _profiles = customer_profile_service or CustomerProfileService(
        db_path=PROJECT_ROOT / ".opentulpa" / "customer_profiles.db"
    )
    _file_vault = file_vault_service or FileVaultService(
        root_dir=PROJECT_ROOT / ".opentulpa" / "file_vault",
        db_path=PROJECT_ROOT / ".opentulpa" / "file_vault.db",
    )
    _skill_store = skill_store_service or SkillStoreService(
        db_path=PROJECT_ROOT / ".opentulpa" / "skills.db",
        root_dir=PROJECT_ROOT / ".opentulpa" / "skills",
    )
    _skill_store.ensure_default_skill()

    settings = get_settings()
    _telegram_client = (
        TelegramClient(settings.telegram_bot_token) if settings.telegram_bot_token else None
    )
    _telegram_chat = (
        TelegramChatService(
            bot_token=settings.telegram_bot_token,
            file_vault=_file_vault,
            memory=_memory,
        )
        if settings.telegram_bot_token
        else None
    )

    async def _process_wake_event(body: dict[str, Any]) -> None:
        logger.info("Processing wake event: %s", body)
        wake_type = str(body.get("type", "")).strip()
        if wake_type not in {"task_event", "routine_event"}:
            return
        if wake_type == "task_event":
            customer_id = str(body.get("customer_id", "")).strip()
            event_type = str(body.get("event_type", "")).strip()
            payload = body.get("payload") if isinstance(body.get("payload"), dict) else {}
            if not customer_id or event_type not in {"done", "failed", "needs_input", "worker_stopped"}:
                return

            should_notify = event_type == "needs_input"
            if not should_notify and _agent_runtime and hasattr(_agent_runtime, "classify_wake_event"):
                decision = await _agent_runtime.classify_wake_event(
                    customer_id=customer_id,
                    event_label=f"task/{event_type}",
                    payload={
                        "task_id": str(body.get("task_id", "")),
                        "payload": payload,
                    },
                )
                should_notify = bool(decision.get("notify_user", False))

            if not should_notify:
                _get_context_events().add_event(
                    customer_id=customer_id,
                    source="task",
                    event_type=event_type,
                    payload={
                        "task_id": str(body.get("task_id", "")),
                        **payload,
                    },
                )
                return

            if not settings.telegram_bot_token:
                _get_context_events().add_event(
                    customer_id=customer_id,
                    source="task",
                    event_type=event_type,
                    payload={
                        "task_id": str(body.get("task_id", "")),
                        **payload,
                    },
                )
                return

            try:
                replies = await _get_telegram_chat().relay_task_event(
                    customer_id=customer_id,
                    task_id=str(body.get("task_id", "")),
                    event_type=event_type,
                    payload=payload,
                    agent_runtime=_agent_runtime,
                )
            except Exception:
                _get_context_events().add_event(
                    customer_id=customer_id,
                    source="task",
                    event_type=event_type,
                    payload={
                        "task_id": str(body.get("task_id", "")),
                        **payload,
                    },
                )
                return
            if not replies:
                _get_context_events().add_event(
                    customer_id=customer_id,
                    source="task",
                    event_type=event_type,
                    payload={
                        "task_id": str(body.get("task_id", "")),
                        **payload,
                    },
                )
                return
            for item in replies:
                await _get_telegram_client().send_message(
                    chat_id=item["chat_id"],
                    text=item["text"],
                    parse_mode="HTML",
                )
            return

        payload = body.get("payload") if isinstance(body.get("payload"), dict) else {}
        customer_id = str(body.get("customer_id") or payload.get("customer_id") or "").strip()
        if not customer_id:
            return
        event_type = str(body.get("event_type") or payload.get("event_type") or "scheduled").strip()
        notify_raw = body.get("notify_user", payload.get("notify_user", True))
        notify_user = not (
            notify_raw is False or str(notify_raw).strip().lower() in {"0", "false", "no", "off"}
        )
        routine_id = str(body.get("routine_id") or payload.get("routine_id") or "").strip()
        routine_name = str(body.get("routine_name") or payload.get("routine_name") or "").strip()
        queue_payload = {
            "routine_id": routine_id,
            "routine_name": routine_name,
            "event_type": event_type,
            "notify_user": bool(notify_user),
            "payload": payload,
        }

        if not notify_user:
            _get_context_events().add_event(
                customer_id=customer_id,
                source="routine",
                event_type=event_type,
                payload=queue_payload,
            )
            return

        if not settings.telegram_bot_token or _agent_runtime is None:
            _get_context_events().add_event(
                customer_id=customer_id,
                source="routine",
                event_type=event_type,
                payload=queue_payload,
            )
            return

        try:
            replies = await _get_telegram_chat().relay_event(
                customer_id=customer_id,
                event_label=f"routine/{event_type}",
                payload=queue_payload,
                agent_runtime=_agent_runtime,
            )
        except Exception:
            _get_context_events().add_event(
                customer_id=customer_id,
                source="routine",
                event_type=event_type,
                payload=queue_payload,
            )
            return
        if not replies:
            _get_context_events().add_event(
                customer_id=customer_id,
                source="routine",
                event_type=event_type,
                payload=queue_payload,
            )
            return
        for item in replies:
            await _get_telegram_client().send_message(
                chat_id=item["chat_id"],
                text=item["text"],
                parse_mode="HTML",
            )

    _wake_queue = WakeQueueService(
        db_path=PROJECT_ROOT / ".opentulpa" / "wake_events.db",
        handler=_process_wake_event,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Ensure tulpa router discovery runs at startup.
        result = _get_tulpa_loader().reload()
        if result.get("errors"):
            logger.warning("Tulpa router load had errors: %s", result["errors"])
        if _agent_runtime and hasattr(_agent_runtime, "start"):
            await _agent_runtime.start()
        if _scheduler:
            _scheduler.start()
        if _tasks:
            await _tasks.start()
        if _wake_queue:
            await _wake_queue.start()
        yield
        if _scheduler:
            _scheduler.shutdown(wait=True)
        if _tasks:
            await _tasks.shutdown()
        if _wake_queue:
            await _wake_queue.shutdown()
        if _agent_runtime and hasattr(_agent_runtime, "shutdown"):
            await _agent_runtime.shutdown()

    app = FastAPI(
        title="OpenTulpa",
        description="Background-capable agent API",
        version="0.1.0",
        lifespan=lifespan,
    )
    _tulpa_loader = TulpaRouterLoader(project_root=PROJECT_ROOT, mount_router=_tulpa_router)
    app.include_router(_tulpa_router, prefix="/tulpa")

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/agent/healthz")
    async def agent_health() -> dict[str, Any]:
        healthy = bool(_agent_runtime and getattr(_agent_runtime, "healthy", lambda: False)())
        return {"status": "ok" if healthy else "degraded", "backend": "langgraph"}

    # ---------- Internal API (for LangGraph tools) ----------
    @app.post("/internal/memory/add")
    async def internal_memory_add(request: Request) -> Any:
        mem = _get_memory()
        body = await request.json()
        messages = body.get("messages", [])
        user_id = body.get("user_id") or mem.user_id
        metadata = body.get("metadata") or {}
        result = mem.add(messages, user_id=user_id, metadata=metadata)
        return {"ok": True, "result": result}

    @app.post("/internal/memory/search")
    async def internal_memory_search(request: Request) -> Any:
        mem = _get_memory()
        body = await request.json()
        query = body.get("query", "")
        user_id = body.get("user_id") or mem.user_id
        limit = body.get("limit", 5)
        results = mem.search(query, user_id=user_id, limit=limit)
        return {"results": results}

    @app.post("/internal/files/search")
    async def internal_files_search(request: Request) -> Any:
        vault = _get_file_vault()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        query = str(body.get("query", "")).strip()
        limit = int(body.get("limit", 5))
        results = [
            _sanitize_uploaded_file_record(r, include_excerpt=False)
            for r in vault.search(customer_id, query=query, limit=limit)
        ]
        return {"ok": True, "results": results}

    @app.post("/internal/files/get")
    async def internal_files_get(request: Request) -> Any:
        vault = _get_file_vault()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        file_id = str(body.get("file_id", "")).strip()
        max_excerpt_chars = max(500, min(int(body.get("max_excerpt_chars", 16000)), 60000))
        record = vault.get_file(customer_id, file_id)
        if not record:
            return JSONResponse(status_code=404, content={"detail": "file not found"})
        return {
            "ok": True,
            "file": _sanitize_uploaded_file_record(
                record,
                include_excerpt=True,
                max_excerpt_chars=max_excerpt_chars,
            ),
        }

    @app.post("/internal/files/send")
    async def internal_files_send(request: Request) -> Any:
        vault = _get_file_vault()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        file_id = str(body.get("file_id", "")).strip()
        caption_raw = body.get("caption")
        caption = str(caption_raw).strip() if caption_raw is not None else None
        caption = caption or None
        if not settings.telegram_bot_token:
            return JSONResponse(status_code=501, content={"detail": "Telegram not configured"})
        if not customer_id or not file_id:
            return JSONResponse(
                status_code=400, content={"detail": "customer_id and file_id are required"}
            )
        record = vault.get_file(customer_id, file_id)
        if not record:
            return JSONResponse(status_code=404, content={"detail": "file not found"})

        chat_id = record.get("chat_id")
        if chat_id is None:
            slots = _get_telegram_chat().find_session_slots(customer_id)
            if slots:
                chat_id = slots[0].get("chat_id")
        if chat_id is None:
            return JSONResponse(status_code=404, content={"detail": "no chat found for customer"})

        raw_bytes = vault.read_file_bytes(customer_id, file_id)
        if raw_bytes is None:
            return JSONResponse(status_code=404, content={"detail": "stored file bytes not found"})

        sent = await _get_telegram_client().send_file(
            chat_id=chat_id,
            filename=str(record.get("original_filename", "file.bin")),
            raw_bytes=raw_bytes,
            kind=str(record.get("kind", "document")),
            mime_type=str(record.get("mime_type", "")).strip() or None,
            caption=caption,
            parse_mode="HTML",
        )
        if not sent:
            return JSONResponse(status_code=502, content={"detail": "telegram send failed"})
        return {"ok": True, "file_id": file_id, "chat_id": chat_id}

    @app.post("/internal/files/send_web_image")
    async def internal_files_send_web_image(request: Request) -> Any:
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        image_url = str(body.get("url", "")).strip()
        caption_raw = body.get("caption")
        caption = str(caption_raw).strip() if caption_raw is not None else None
        caption = caption or None
        max_bytes = int(body.get("max_bytes", 10_000_000))

        if not settings.telegram_bot_token:
            return JSONResponse(status_code=501, content={"detail": "Telegram not configured"})
        if not customer_id or not image_url:
            return JSONResponse(
                status_code=400,
                content={"detail": "customer_id and url are required"},
            )

        chat_id: Any = None
        slots = _get_telegram_chat().find_session_slots(customer_id)
        if slots:
            chat_id = slots[0].get("chat_id")
        if chat_id is None:
            return JSONResponse(status_code=404, content={"detail": "no chat found for customer"})

        try:
            downloaded = await _download_image_from_web_url(
                image_url,
                max_bytes=max_bytes,
            )
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=502, content={"detail": f"image fetch failed: {exc}"})

        sent = await _get_telegram_client().send_file(
            chat_id=chat_id,
            filename=str(downloaded["filename"]),
            raw_bytes=downloaded["raw_bytes"],
            kind="animation" if str(downloaded["content_type"]).strip().lower() == "image/gif" else "photo",
            mime_type=str(downloaded["content_type"]),
            caption=caption,
            parse_mode="HTML",
        )
        if not sent:
            return JSONResponse(status_code=502, content={"detail": "telegram send failed"})
        return {
            "ok": True,
            "chat_id": chat_id,
            "url": str(downloaded["final_url"]),
            "mime_type": str(downloaded["content_type"]),
            "size_bytes": int(downloaded["size_bytes"]),
        }

    @app.post("/internal/files/analyze")
    async def internal_files_analyze(request: Request) -> Any:
        vault = _get_file_vault()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        file_id = str(body.get("file_id", "")).strip()
        question_raw = body.get("question")
        question = str(question_raw).strip() if question_raw is not None else None
        question = question or None
        if not customer_id or not file_id:
            return JSONResponse(
                status_code=400, content={"detail": "customer_id and file_id are required"}
            )
        if _agent_runtime is None or not hasattr(_agent_runtime, "analyze_uploaded_file"):
            return JSONResponse(
                status_code=501,
                content={"detail": "agent runtime unavailable for file analysis"},
            )
        record = vault.get_file(customer_id, file_id)
        if not record:
            return JSONResponse(status_code=404, content={"detail": "file not found"})
        raw_bytes = vault.read_file_bytes(customer_id, file_id)
        if raw_bytes is None:
            return JSONResponse(status_code=404, content={"detail": "stored file bytes not found"})
        try:
            analysis_result = await _agent_runtime.analyze_uploaded_file(
                record=record,
                raw_bytes=raw_bytes,
                question=question,
            )
        except Exception as exc:
            return JSONResponse(status_code=500, content={"detail": f"file analysis failed: {exc}"})

        if not question:
            analysis_text = str(analysis_result.get("analysis", "")).strip()
            if analysis_text:
                updated = vault.set_ai_summary(customer_id, file_id, analysis_text)
                if isinstance(updated, dict):
                    record = updated
        return {
            "ok": True,
            "analysis": str(analysis_result.get("analysis", "")).strip(),
            "file": _sanitize_uploaded_file_record(record, include_excerpt=True, max_excerpt_chars=16000),
        }

    @app.post("/internal/directive/get")
    async def internal_directive_get(request: Request) -> Any:
        profiles = _get_profiles()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        return {
            "customer_id": customer_id,
            "directive": profiles.get_directive(customer_id),
        }

    @app.post("/internal/directive/set")
    async def internal_directive_set(request: Request) -> Any:
        profiles = _get_profiles()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        directive = str(body.get("directive", "")).strip()
        source = str(body.get("source", "agent") or "agent")
        profiles.set_directive(customer_id, directive, source=source)

        # Best-effort memory signal for recall; directive DB remains source of truth.
        if _memory is not None:
            with suppress(Exception):
                _memory.add_text(
                    f"Directive updated for this user: {directive}",
                    user_id=customer_id,
                    metadata={"kind": "directive_profile", "source": source},
                )

        return {"ok": True, "customer_id": customer_id}

    @app.post("/internal/directive/clear")
    async def internal_directive_clear(request: Request) -> Any:
        profiles = _get_profiles()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        cleared = profiles.clear_directive(customer_id, source="agent")

        # Best-effort memory signal for recall; directive DB remains source of truth.
        if _memory is not None:
            with suppress(Exception):
                _memory.add_text(
                    "Directive profile cleared for this user. Previous directive no longer applies.",
                    user_id=customer_id,
                    metadata={"kind": "directive_profile", "source": "agent"},
                )

        return {"ok": True, "customer_id": customer_id, "cleared": cleared}

    @app.post("/internal/time_profile/get")
    async def internal_time_profile_get(request: Request) -> Any:
        profiles = _get_profiles()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        return {
            "customer_id": customer_id,
            "utc_offset": profiles.get_utc_offset(customer_id),
        }

    @app.post("/internal/time_profile/set")
    async def internal_time_profile_set(request: Request) -> Any:
        profiles = _get_profiles()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        utc_offset = str(body.get("utc_offset", "")).strip()
        source = str(body.get("source", "agent") or "agent")
        normalized = profiles.set_utc_offset(customer_id, utc_offset, source=source)
        return {"ok": True, "customer_id": customer_id, "utc_offset": normalized}

    @app.post("/internal/skills/list")
    async def internal_skills_list(request: Request) -> Any:
        store = _get_skill_store()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        include_global = bool(body.get("include_global", True))
        include_disabled = bool(body.get("include_disabled", False))
        limit = int(body.get("limit", 200))
        skills = store.list_skills(
            customer_id=customer_id,
            include_global=include_global,
            include_disabled=include_disabled,
            limit=limit,
        )
        return {"ok": True, "skills": skills}

    @app.post("/internal/skills/get")
    async def internal_skills_get(request: Request) -> Any:
        store = _get_skill_store()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        name = str(body.get("name", "")).strip()
        include_files = bool(body.get("include_files", True))
        include_global = bool(body.get("include_global", True))
        if not name:
            return JSONResponse(status_code=400, content={"detail": "name is required"})
        skill = store.get_skill(
            customer_id=customer_id,
            name=name,
            include_files=include_files,
            include_global=include_global,
        )
        if skill is None:
            return JSONResponse(status_code=404, content={"detail": "skill not found"})
        return {"ok": True, "skill": skill}

    @app.post("/internal/skills/upsert")
    async def internal_skills_upsert(request: Request) -> Any:
        store = _get_skill_store()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        scope = str(body.get("scope", "user")).strip().lower()
        name = str(body.get("name", "")).strip()
        description = str(body.get("description", "")).strip()
        instructions = str(body.get("instructions", "")).strip()
        skill_markdown = str(body.get("skill_markdown", "")).strip()
        source = str(body.get("source", "agent") or "agent")
        supporting_files_raw = body.get("supporting_files")
        supporting_files = (
            supporting_files_raw if isinstance(supporting_files_raw, dict) else None
        )
        if scope == "user" and not customer_id:
            return JSONResponse(
                status_code=400, content={"detail": "customer_id is required for user skills"}
            )
        if not name:
            return JSONResponse(status_code=400, content={"detail": "name is required"})
        try:
            if not skill_markdown:
                from opentulpa.skills.service import build_skill_markdown

                skill_markdown = build_skill_markdown(
                    name=name,
                    description=description,
                    instructions=instructions,
                )
            skill = store.upsert_skill(
                scope=scope,
                customer_id=customer_id,
                name=name,
                skill_markdown=skill_markdown,
                source=source,
                enabled=True,
                supporting_files=supporting_files,
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})

        if _memory is not None:
            with suppress(Exception):
                _memory.add_text(
                    (
                        "Skill stored for this user: "
                        f"name={skill.get('name')} scope={skill.get('scope')} "
                        f"description={skill.get('description')}"
                    ),
                    user_id=customer_id or "global",
                    metadata={
                        "kind": "user_skill",
                        "skill_name": skill.get("name"),
                        "scope": skill.get("scope"),
                    },
                )
        return {"ok": True, "skill": skill}

    @app.post("/internal/skills/delete")
    async def internal_skills_delete(request: Request) -> Any:
        store = _get_skill_store()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        scope = str(body.get("scope", "user")).strip().lower()
        name = str(body.get("name", "")).strip()
        if not name:
            return JSONResponse(status_code=400, content={"detail": "name is required"})
        if scope == "user" and not customer_id:
            return JSONResponse(
                status_code=400, content={"detail": "customer_id is required for user skills"}
            )
        try:
            deleted = store.delete_skill(scope=scope, customer_id=customer_id, name=name)
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return {"ok": True, "deleted": bool(deleted)}

    @app.post("/internal/scheduler/routine")
    async def internal_scheduler_add_routine(request: Request) -> Any:
        sched = _get_scheduler()
        body = await request.json()

        from opentulpa.scheduler.models import Routine

        rid = str(body.get("id", "")).strip() or new_short_id("rtn")
        routine = Routine(
            id=rid,
            name=body.get("name", "Unnamed"),
            schedule=body.get("schedule", "0 9 * * *"),
            payload=body.get("payload", {}),
            enabled=body.get("enabled", True),
            is_cron=body.get("is_cron", True),
        )
        sched.add_routine(routine)
        return {"ok": True, "id": rid}

    @app.get("/internal/scheduler/routines")
    async def internal_scheduler_list_routines(customer_id: str | None = None) -> Any:
        sched = _get_scheduler()
        routines = sched.list_routines()
        cid = str(customer_id or "").strip()
        if cid:
            routines = [
                r
                for r in routines
                if str((r.payload or {}).get("customer_id", "")).strip() == cid
            ]
        return {
            "routines": [
                {
                    "id": r.id,
                    "name": r.name,
                    "schedule": r.schedule,
                    "enabled": r.enabled,
                    "is_cron": r.is_cron,
                }
                for r in routines
            ]
        }

    @app.delete("/internal/scheduler/routine/{routine_id}")
    async def internal_scheduler_remove_routine(
        routine_id: str,
        customer_id: str | None = None,
    ) -> Any:
        sched = _get_scheduler()
        cid = str(customer_id or "").strip()
        if cid:
            routine = sched.get_routine(routine_id)
            if routine is None:
                return {"ok": False}
            owner = str((routine.payload or {}).get("customer_id", "")).strip()
            if owner != cid:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "routine does not belong to this customer_id"},
                )
        ok = sched.remove_routine(routine_id)
        return {"ok": ok}

    @app.post("/internal/scheduler/routine/delete_with_assets")
    async def internal_scheduler_remove_routine_with_assets(request: Request) -> Any:
        sched = _get_scheduler()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        if not customer_id:
            return JSONResponse(status_code=400, content={"detail": "customer_id is required"})

        routine_id = str(body.get("routine_id", "")).strip()
        name = str(body.get("name", "")).strip()
        remove_all_matches = bool(body.get("remove_all_matches", False))
        delete_files = bool(body.get("delete_files", True))
        extra_cleanup_paths = _normalize_cleanup_paths(body.get("cleanup_paths"))

        routines = [
            r
            for r in sched.list_routines()
            if str((r.payload or {}).get("customer_id", "")).strip() == customer_id
        ]
        if routine_id:
            matched = [r for r in routines if r.id == routine_id]
        elif name:
            name_cf = name.strip().casefold()
            matched = [r for r in routines if r.name.strip().casefold() == name_cf]
        else:
            return JSONResponse(
                status_code=400,
                content={"detail": "routine_id or name is required"},
            )

        if not matched:
            return JSONResponse(status_code=404, content={"detail": "routine not found"})
        if len(matched) > 1 and not remove_all_matches:
            return JSONResponse(
                status_code=400,
                content={
                    "detail": "multiple routines matched; set remove_all_matches=true",
                    "matched_routines": [
                        {"id": r.id, "name": r.name, "schedule": r.schedule} for r in matched
                    ],
                },
            )

        deleted_routines: list[dict[str, Any]] = []
        failed_routines: list[dict[str, Any]] = []
        deleted_files: list[dict[str, Any]] = []
        failed_files: list[dict[str, Any]] = []

        for routine in matched:
            ok = sched.remove_routine(routine.id)
            if not ok:
                failed_routines.append({"id": routine.id, "name": routine.name, "error": "not found"})
                continue
            deleted_routines.append({"id": routine.id, "name": routine.name})
            if not delete_files:
                continue

            cleanup_paths = _collect_routine_cleanup_paths(routine.payload or {})
            cleanup_paths.extend(extra_cleanup_paths)
            seen_paths: set[str] = set()
            unique_paths: list[str] = []
            for path in cleanup_paths:
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                unique_paths.append(path)

            for relative_path in unique_paths:
                try:
                    result = sandbox_delete_file(relative_path, missing_ok=True)
                    deleted_files.append(
                        {
                            "path": str(result.get("path", relative_path)),
                            "deleted": bool(result.get("deleted", False)),
                            "missing": bool(result.get("missing", False)),
                        }
                    )
                except Exception as exc:
                    failed_files.append({"path": relative_path, "error": str(exc)})

        return {
            "ok": len(deleted_routines) > 0 and len(failed_routines) == 0,
            "deleted_routines": deleted_routines,
            "failed_routines": failed_routines,
            "deleted_files": deleted_files,
            "failed_files": failed_files,
        }

    @app.post("/internal/wake")
    async def internal_wake(request: Request) -> Any:
        """Called by scheduler or external trigger to wake the agent with a payload."""
        body = await request.json()
        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400, content={"detail": "wake payload must be JSON object"}
            )
        queue_id = await _get_wake_queue().enqueue(body)
        logger.info("Wake requested and queued id=%s payload=%s", queue_id, body)
        return {"ok": True, "queued": True, "queue_id": queue_id}

    @app.get("/internal/wake/queue")
    async def internal_wake_queue_stats() -> Any:
        """Inspect wake queue health and recent entries."""
        return {"ok": True, "queue": _get_wake_queue().stats()}

    @app.post("/internal/web_search")
    async def internal_web_search(request: Request) -> Any:
        """Run OpenRouter web search (:online). Returns model response with current web context."""
        body = await request.json()
        query = body.get("query", "").strip()
        if not query:
            return JSONResponse(status_code=400, content={"detail": "query required"})
        model = f"google/{settings.llm_model}" if settings.llm_model else None
        result = await run_web_search(query, model=model)
        return {"ok": True, "result": result}

    @app.post("/internal/tulpa/reload")
    async def internal_tulpa_reload() -> Any:
        """Reload APIRouter modules from tulpa_stuff."""
        return _get_tulpa_loader().reload()

    @app.post("/internal/tulpa/write_file")
    async def internal_tulpa_write_file(request: Request) -> Any:
        """Write a file only inside approved integration/self-modification paths."""
        body = await request.json()
        relative_path = str(body.get("path", "")).strip()
        content = body.get("content")
        if content is None:
            return JSONResponse(status_code=400, content={"detail": "content is required"})
        try:
            target = sandbox_write_file(relative_path, str(content))
            validation = sandbox_validate_generated_file(relative_path)
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return {
            "ok": True,
            "path": str(target.relative_to(PROJECT_ROOT)),
            "validation": validation,
        }

    @app.post("/internal/tulpa/validate_file")
    async def internal_tulpa_validate_file(request: Request) -> Any:
        """Validate generated code file contract/syntax before using it."""
        body = await request.json()
        relative_path = str(body.get("path", "")).strip()
        try:
            result = sandbox_validate_generated_file(relative_path)
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return result

    @app.get("/internal/tulpa/read_file")
    async def internal_tulpa_read_file(path: str, max_chars: int = 12000) -> Any:
        """Read a file inside approved integration/self-modification paths."""
        try:
            content = sandbox_read_file(path, max_chars=max_chars)
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return {"ok": True, "path": path, "content": content}

    @app.post("/internal/tulpa/run_terminal")
    async def internal_tulpa_run_terminal(request: Request) -> Any:
        """Run a restricted command in approved integration/self-modification paths."""
        body = await request.json()
        command = str(body.get("command", "")).strip()
        working_dir_key = str(body.get("working_dir", "tulpa_stuff")).strip()
        timeout_seconds = int(body.get("timeout_seconds", 90))

        if not command:
            return JSONResponse(status_code=400, content={"detail": "command is required"})
        try:
            return sandbox_run_terminal(
                command=command,
                working_dir=working_dir_key,
                timeout_seconds=timeout_seconds,
            )
        except PermissionError as exc:
            return JSONResponse(
                status_code=403,
                content={"detail": str(exc), "allowed_commands": sorted(ALLOWED_TERMINAL_COMMANDS)},
            )
        except TimeoutError as exc:
            return JSONResponse(status_code=408, content={"detail": str(exc)})
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={
                    "detail": str(exc),
                    "allowed_working_dirs": sorted(ALLOWED_TERMINAL_DIRS.keys()),
                },
            )
        except RuntimeError as exc:
            return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.get("/internal/tulpa/catalog")
    async def internal_tulpa_catalog() -> Any:
        """Return tulpa_stuff catalog/index and recent tracked entries."""
        return {"ok": True, "catalog": get_tulpa_catalog()}

    # ---------- Task orchestration API ----------
    @app.post("/internal/tasks/create")
    async def internal_task_create(request: Request) -> Any:
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        goal = str(body.get("goal", "")).strip()
        payload = body.get("payload") or {}
        risk_level = str(body.get("risk_level", "low")).strip() or "low"
        idempotency_key = body.get("idempotency_key")
        if not customer_id or not goal:
            return JSONResponse(
                status_code=400, content={"detail": "customer_id and goal are required"}
            )
        if isinstance(payload, dict) and "steps" in payload:
            steps = payload.get("steps")
            if not isinstance(steps, list):
                return JSONResponse(
                    status_code=400, content={"detail": "payload.steps must be a list"}
                )
            bad_idx = next((i for i, s in enumerate(steps) if not isinstance(s, dict)), None)
            if bad_idx is not None:
                return JSONResponse(
                    status_code=400,
                    content={
                        "detail": (
                            "payload.steps entries must be objects with a 'type' field "
                            "(e.g. {'type':'run_terminal', ...})."
                        ),
                        "bad_step_index": bad_idx,
                    },
                )
        task = await _get_tasks().create_task(
            customer_id=customer_id,
            goal=goal,
            payload=payload,
            risk_level=risk_level,
            idempotency_key=idempotency_key,
        )
        return {"ok": True, "task": task}

    @app.get("/internal/tasks/{task_id}")
    async def internal_task_status(task_id: str) -> Any:
        try:
            task = _get_tasks().get_task(task_id)
        except KeyError:
            return JSONResponse(status_code=404, content={"detail": "task not found"})
        return {"ok": True, "task": task}

    @app.get("/internal/tasks/{task_id}/events")
    async def internal_task_events(task_id: str, limit: int = 50, offset: int = 0) -> Any:
        try:
            _get_tasks().get_task(task_id)
        except KeyError:
            return JSONResponse(status_code=404, content={"detail": "task not found"})
        events = _get_tasks().list_events(task_id, limit=limit, offset=offset)
        return {"ok": True, "events": events}

    @app.get("/internal/tasks/{task_id}/artifacts")
    async def internal_task_artifacts(task_id: str) -> Any:
        try:
            _get_tasks().get_task(task_id)
        except KeyError:
            return JSONResponse(status_code=404, content={"detail": "task not found"})
        return {"ok": True, "artifacts": _get_tasks().list_task_artifacts(task_id)}

    @app.post("/internal/tasks/{task_id}/relaunch")
    async def internal_task_relaunch(task_id: str, request: Request) -> Any:
        body = await request.json()
        clarification = body.get("clarification")
        trigger_reason = (
            str(body.get("trigger_reason", "user_requested")).strip() or "user_requested"
        )
        try:
            task = await _get_tasks().relaunch_task(
                task_id=task_id,
                trigger_reason=trigger_reason,
                clarification=clarification,
            )
        except KeyError:
            return JSONResponse(status_code=404, content={"detail": "task not found"})
        return {"ok": True, "task": task}

    @app.post("/internal/tasks/{task_id}/cancel")
    async def internal_task_cancel(task_id: str) -> Any:
        try:
            task = await _get_tasks().cancel_task(task_id)
        except KeyError:
            return JSONResponse(status_code=404, content={"detail": "task not found"})
        return {"ok": True, "task": task}

    # ---------- Internal Slack API (for LangGraph Slack tools) ----------
    @app.get("/internal/slack/channels")
    async def internal_slack_channels(limit: int = 100, cursor: str = "") -> Any:
        if _slack is None:
            return JSONResponse(status_code=501, content={"detail": "Slack not configured"})
        result = await _get_slack().list_channels(limit=limit, cursor=cursor or None)
        return result

    @app.get("/internal/slack/channels/{channel_id}/history")
    async def internal_slack_history(channel_id: str, limit: int = 20, cursor: str = "") -> Any:
        if _slack is None:
            return JSONResponse(status_code=501, content={"detail": "Slack not configured"})
        result = await _get_slack().channel_history(channel_id, limit=limit, cursor=cursor or None)
        return result

    @app.post("/internal/slack/consent")
    async def internal_slack_consent(request: Request) -> Any:
        """Grant Slack write consent for this customer (called when user confirms in chat)."""
        body = await request.json()
        customer_id = body.get("customer_id")
        scope = body.get("scope", "write")
        if not customer_id:
            return JSONResponse(status_code=400, content={"detail": "customer_id required"})
        if scope != "write":
            return JSONResponse(status_code=400, content={"detail": "scope must be 'write'"})
        grant_slack_write_consent(customer_id)
        return {"ok": True, "message": "Slack write consent granted."}

    @app.post("/internal/slack/post")
    async def internal_slack_post(request: Request) -> Any:
        if _slack is None:
            return JSONResponse(status_code=501, content={"detail": "Slack not configured"})
        body = await request.json()
        customer_id = body.get("customer_id")
        channel_id = body.get("channel_id", "")
        text = body.get("text", "")
        thread_ts = body.get("thread_ts")
        if not customer_id:
            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error": "customer_id required",
                    "detail": "customer_id required",
                },
            )
        if not channel_id or not text:
            return JSONResponse(status_code=400, content={"detail": "channel_id and text required"})
        if not has_slack_write_consent(customer_id):
            return JSONResponse(
                status_code=403,
                content={
                    "ok": False,
                    "error": "consent_required",
                    "message": (
                        "The user has not granted permission to post to Slack. "
                        "Ask the user to confirm they allow the agent to post to Slack on their behalf; "
                        "once they agree, use slack_grant_write_consent and then try posting again."
                    ),
                },
            )
        result = await _get_slack().post_message(channel_id, text, thread_ts=thread_ts)
        return result

    # ---------- Telegram webhook ----------
    @app.post("/webhook/telegram")
    async def telegram_webhook(request: Request, background_tasks: BackgroundTasks) -> Response:
        if not settings.telegram_bot_token:
            return JSONResponse(status_code=501, content={"detail": "Telegram not configured"})
        if settings.telegram_webhook_secret:
            incoming_secret = request.headers.get("x-telegram-bot-api-secret-token", "")
            if incoming_secret != settings.telegram_webhook_secret:
                return JSONResponse(status_code=403, content={"detail": "invalid telegram secret"})
        body = await request.json()

        # Immediate 200 OK, logic runs in background
        background_tasks.add_task(_telegram_background_handler, body=body, settings=settings)
        return Response(status_code=200)

    async def _telegram_background_handler(body: dict, settings: Any):
        message = body.get("message") or body.get("edited_message") or {}
        chat_id = message.get("chat", {}).get("id")
        try:
            reply = await _get_telegram_chat().handle_update(
                body=body,
                allowed_user_ids_csv=settings.telegram_allowed_user_ids,
                allowed_usernames_csv=settings.telegram_allowed_usernames,
                agent_runtime=_agent_runtime,
            )
        except Exception as exc:
            logger.exception("Unhandled Telegram background handler failure: %s", exc)
            if chat_id is not None:
                with suppress(Exception):
                    await _get_telegram_client().send_message(
                        chat_id=chat_id,
                        text="I hit an internal error while processing your message. Please try again.",
                        parse_mode="HTML",
                    )
            return

        if reply and chat_id is not None:
            with suppress(Exception):
                await _get_telegram_client().send_message(
                    chat_id=chat_id,
                    text=reply,
                    parse_mode="HTML",
                )

    return app
