"""FastAPI application: health, internal API, Telegram webhook, and agent runtime."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI

from opentulpa.api.file_helpers import (
    download_image_from_web_url,
    infer_image_filename,
    safe_telegram_filename,
)
from opentulpa.api.routes import (
    register_approval_routes,
    register_file_routes,
    register_health_routes,
    register_memory_routes,
    register_profile_routes,
    register_scheduler_routes,
    register_skill_routes,
    register_slack_routes,
    register_task_routes,
    register_telegram_webhook_routes,
    register_tulpa_routes,
    register_wake_and_search_routes,
)
from opentulpa.api.tulpa_loader import TulpaRouterLoader
from opentulpa.approvals.adapters.telegram import TelegramApprovalAdapter
from opentulpa.approvals.adapters.text_token import TextTokenApprovalAdapter
from opentulpa.approvals.broker import ApprovalBroker
from opentulpa.approvals.store import PendingApprovalStore
from opentulpa.context.customer_profiles import CustomerProfileService
from opentulpa.context.file_vault import FileVaultService
from opentulpa.context.service import EventContextService
from opentulpa.core.config import get_settings
from opentulpa.integrations.slack_client import grant_slack_write_consent, has_slack_write_consent
from opentulpa.interfaces.telegram.chat_service import TelegramChatService
from opentulpa.interfaces.telegram.client import TelegramClient
from opentulpa.memory.service import MemoryService
from opentulpa.scheduler.service import SchedulerService
from opentulpa.skills.service import SkillStoreService
from opentulpa.tasks.sandbox import (
    PROJECT_ROOT,
)
from opentulpa.tasks.sandbox import delete_file as sandbox_delete_file
from opentulpa.tasks.service import TaskService
from opentulpa.tasks.wake_queue import WakeQueueService

logger = logging.getLogger(__name__)

# Backward-compatible helper exports for existing tests/callers.
_download_image_from_web_url = download_image_from_web_url
_infer_image_filename = infer_image_filename
_safe_telegram_filename = safe_telegram_filename

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
_approval_store: PendingApprovalStore | None = None
_approvals: ApprovalBroker | None = None
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


def _get_approvals() -> ApprovalBroker:
    if _approvals is None:
        raise RuntimeError("ApprovalBroker not initialized")
    return _approvals


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
    global _memory, _scheduler, _slack, _tulpa_loader, _tasks, _wake_queue, _agent_runtime, _context_events, _profiles, _file_vault, _skill_store, _telegram_chat, _telegram_client, _approval_store, _approvals
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

    def _resolve_approval_origin(customer_id: str, thread_id: str) -> dict[str, Any]:
        if _telegram_chat is None:
            return {}
        slots = _telegram_chat.find_session_slots(customer_id)
        if not slots:
            return {}
        selected = None
        safe_thread = str(thread_id or "").strip()
        for slot in slots:
            if safe_thread and safe_thread in {
                str(slot.get("thread_id", "")).strip(),
                str(slot.get("wake_thread_id", "")).strip(),
            }:
                selected = slot
                break
        if selected is None:
            selected = slots[0]
        chat_id = str(selected.get("chat_id", "")).strip()
        user_id = str(selected.get("user_id", "")).strip()
        if not chat_id:
            return {}
        return {
            "origin_interface": "telegram",
            "origin_user_id": user_id,
            "origin_conversation_id": chat_id,
        }

    approval_db = Path(settings.approvals_db_path)
    if not approval_db.is_absolute():
        approval_db = (PROJECT_ROOT / approval_db).resolve()
    _approval_store = PendingApprovalStore(db_path=approval_db)
    telegram_adapter = (
        TelegramApprovalAdapter(client=_telegram_client) if _telegram_client is not None else None
    )
    text_token_adapter = TextTokenApprovalAdapter(telegram_client=_telegram_client)
    _approvals = ApprovalBroker(
        store=_approval_store,
        runtime=_agent_runtime,
        approval_ttl_minutes=settings.approvals_ttl_minutes,
        adapters={"telegram": telegram_adapter} if telegram_adapter is not None else {},
        text_token_adapter=text_token_adapter,
        origin_resolver=_resolve_approval_origin,
    )

    async def _process_wake_event(body: dict[str, Any]) -> None:
        logger.info("Processing wake event: %s", body)
        wake_type = str(body.get("type", "")).strip()
        if wake_type not in {"task_event", "routine_event", "approval_event"}:
            return
        if wake_type == "approval_event":
            customer_id = str(body.get("customer_id", "")).strip()
            payload = body.get("payload") if isinstance(body.get("payload"), dict) else {}
            event_type = str(body.get("event_type", payload.get("event_type", "approved"))).strip()
            if not customer_id:
                return
            queue_payload = {
                "approval_id": str(body.get("approval_id", payload.get("approval_id", ""))).strip(),
                "event_type": event_type,
                "payload": payload,
            }
            if not settings.telegram_bot_token or _agent_runtime is None:
                _get_context_events().add_event(
                    customer_id=customer_id,
                    source="approval",
                    event_type=event_type,
                    payload=queue_payload,
                )
                return
            try:
                replies = await _get_telegram_chat().relay_event(
                    customer_id=customer_id,
                    event_label=f"approval/{event_type}",
                    payload=queue_payload,
                    agent_runtime=_agent_runtime,
                )
            except Exception:
                _get_context_events().add_event(
                    customer_id=customer_id,
                    source="approval",
                    event_type=event_type,
                    payload=queue_payload,
                )
                return
            if not replies:
                _get_context_events().add_event(
                    customer_id=customer_id,
                    source="approval",
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
    register_health_routes(app, get_agent_runtime=lambda: _agent_runtime)
    register_memory_routes(app, get_memory=_get_memory)
    register_file_routes(
        app,
        get_file_vault=_get_file_vault,
        get_telegram_chat=_get_telegram_chat,
        get_telegram_client=_get_telegram_client,
        get_agent_runtime=lambda: _agent_runtime,
        telegram_enabled=bool(settings.telegram_bot_token),
    )
    register_profile_routes(
        app,
        get_profiles=_get_profiles,
        get_memory=lambda: _memory,
    )
    register_skill_routes(
        app,
        get_skill_store=_get_skill_store,
        get_memory=lambda: _memory,
    )

    _decide_approval_and_maybe_wake = register_approval_routes(
        app,
        get_approvals=_get_approvals,
        get_wake_queue=_get_wake_queue,
        get_agent_runtime=lambda: _agent_runtime,
    )

    register_scheduler_routes(
        app,
        get_scheduler=_get_scheduler,
        delete_file=sandbox_delete_file,
    )
    register_wake_and_search_routes(
        app,
        get_wake_queue=_get_wake_queue,
        llm_model=settings.llm_model,
    )
    register_tulpa_routes(app, get_tulpa_loader=_get_tulpa_loader)
    register_task_routes(app, get_tasks=_get_tasks)

    if _slack is not None:
        register_slack_routes(
            app,
            get_slack=_get_slack,
            has_write_consent=has_slack_write_consent,
            grant_write_consent=grant_slack_write_consent,
        )

    register_telegram_webhook_routes(
        app,
        settings=settings,
        get_telegram_client=_get_telegram_client,
        get_telegram_chat=_get_telegram_chat,
        get_agent_runtime=lambda: _agent_runtime,
        decide_approval_and_maybe_wake=_decide_approval_and_maybe_wake,
    )

    return app
