"""FastAPI application: health, internal API, Telegram webhook, and agent runtime."""

from __future__ import annotations

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
from opentulpa.tasks.sandbox import PROJECT_ROOT
from opentulpa.tasks.sandbox import delete_file as sandbox_delete_file
from opentulpa.tasks.service import TaskService
from opentulpa.tasks.wake_queue import WakeQueueService

logger = logging.getLogger(__name__)

# Backward-compatible helper exports for existing tests/callers.
_download_image_from_web_url = download_image_from_web_url
_infer_image_filename = infer_image_filename
_safe_telegram_filename = safe_telegram_filename


def _require(value: Any, name: str) -> Any:
    if value is None:
        raise RuntimeError(f"{name} not initialized")
    return value


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
    memory_service = memory
    scheduler_service = scheduler
    slack_service = slack_client
    task_runner = task_service
    runtime = agent_runtime
    context_events_service = context_events or EventContextService(
        db_path=PROJECT_ROOT / ".opentulpa" / "context_events.db"
    )
    profile_service = customer_profile_service or CustomerProfileService(
        db_path=PROJECT_ROOT / ".opentulpa" / "customer_profiles.db"
    )
    vault_service = file_vault_service or FileVaultService(
        root_dir=PROJECT_ROOT / ".opentulpa" / "file_vault",
        db_path=PROJECT_ROOT / ".opentulpa" / "file_vault.db",
    )
    skill_service = skill_store_service or SkillStoreService(
        db_path=PROJECT_ROOT / ".opentulpa" / "skills.db",
        root_dir=PROJECT_ROOT / ".opentulpa" / "skills",
    )
    skill_service.ensure_default_skill()

    settings = get_settings()
    telegram_client = (
        TelegramClient(settings.telegram_bot_token) if settings.telegram_bot_token else None
    )
    telegram_chat = (
        TelegramChatService(
            bot_token=settings.telegram_bot_token,
            file_vault=vault_service,
            memory=memory_service,
        )
        if settings.telegram_bot_token
        else None
    )

    def get_memory() -> MemoryService:
        return _require(memory_service, "MemoryService")

    def get_scheduler() -> SchedulerService:
        return _require(scheduler_service, "SchedulerService")

    def get_slack() -> Any:
        return _require(slack_service, "Slack")

    def get_tasks() -> TaskService:
        return _require(task_runner, "TaskService")

    def get_context_events() -> EventContextService:
        return _require(context_events_service, "EventContextService")

    def get_profiles() -> CustomerProfileService:
        return _require(profile_service, "CustomerProfileService")

    def get_file_vault() -> FileVaultService:
        return _require(vault_service, "FileVaultService")

    def get_skill_store() -> SkillStoreService:
        return _require(skill_service, "SkillStoreService")

    def get_telegram_chat() -> TelegramChatService:
        return _require(telegram_chat, "TelegramChatService")

    def get_telegram_client() -> TelegramClient:
        return _require(telegram_client, "TelegramClient")

    def get_agent_runtime() -> Any:
        return runtime

    def resolve_approval_origin(customer_id: str, thread_id: str) -> dict[str, Any]:
        if telegram_chat is None:
            return {}
        slots = telegram_chat.find_session_slots(customer_id)
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
    approval_store = PendingApprovalStore(db_path=approval_db)
    telegram_adapter = TelegramApprovalAdapter(client=telegram_client) if telegram_client else None
    text_token_adapter = TextTokenApprovalAdapter(telegram_client=telegram_client)
    approvals = ApprovalBroker(
        store=approval_store,
        runtime=runtime,
        approval_ttl_minutes=settings.approvals_ttl_minutes,
        adapters={"telegram": telegram_adapter} if telegram_adapter is not None else {},
        text_token_adapter=text_token_adapter,
        origin_resolver=resolve_approval_origin,
    )

    def get_approvals() -> ApprovalBroker:
        return approvals

    wake_queue_service: WakeQueueService | None = None
    tulpa_loader: TulpaRouterLoader | None = None
    tulpa_router = APIRouter()

    def get_wake_queue() -> WakeQueueService:
        return _require(wake_queue_service, "WakeQueueService")

    def get_tulpa_loader() -> TulpaRouterLoader:
        return _require(tulpa_loader, "TulpaRouterLoader")

    async def process_wake_event(body: dict[str, Any]) -> None:
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
            if not settings.telegram_bot_token or runtime is None:
                get_context_events().add_event(
                    customer_id=customer_id,
                    source="approval",
                    event_type=event_type,
                    payload=queue_payload,
                )
                return
            try:
                replies = await get_telegram_chat().relay_event(
                    customer_id=customer_id,
                    event_label=f"approval/{event_type}",
                    payload=queue_payload,
                    agent_runtime=runtime,
                )
            except Exception:
                get_context_events().add_event(
                    customer_id=customer_id,
                    source="approval",
                    event_type=event_type,
                    payload=queue_payload,
                )
                return
            if not replies:
                get_context_events().add_event(
                    customer_id=customer_id,
                    source="approval",
                    event_type=event_type,
                    payload=queue_payload,
                )
                return
            for item in replies:
                await get_telegram_client().send_message(
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
            if not should_notify and runtime and hasattr(runtime, "classify_wake_event"):
                decision = await runtime.classify_wake_event(
                    customer_id=customer_id,
                    event_label=f"task/{event_type}",
                    payload={
                        "task_id": str(body.get("task_id", "")),
                        "payload": payload,
                    },
                )
                should_notify = bool(decision.get("notify_user", False))

            if not should_notify:
                get_context_events().add_event(
                    customer_id=customer_id,
                    source="task",
                    event_type=event_type,
                    payload={"task_id": str(body.get("task_id", "")), **payload},
                )
                return
            if not settings.telegram_bot_token:
                get_context_events().add_event(
                    customer_id=customer_id,
                    source="task",
                    event_type=event_type,
                    payload={"task_id": str(body.get("task_id", "")), **payload},
                )
                return
            try:
                replies = await get_telegram_chat().relay_task_event(
                    customer_id=customer_id,
                    task_id=str(body.get("task_id", "")),
                    event_type=event_type,
                    payload=payload,
                    agent_runtime=runtime,
                )
            except Exception:
                get_context_events().add_event(
                    customer_id=customer_id,
                    source="task",
                    event_type=event_type,
                    payload={"task_id": str(body.get("task_id", "")), **payload},
                )
                return
            if not replies:
                get_context_events().add_event(
                    customer_id=customer_id,
                    source="task",
                    event_type=event_type,
                    payload={"task_id": str(body.get("task_id", "")), **payload},
                )
                return
            for item in replies:
                await get_telegram_client().send_message(
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
            get_context_events().add_event(
                customer_id=customer_id,
                source="routine",
                event_type=event_type,
                payload=queue_payload,
            )
            return
        if not settings.telegram_bot_token or runtime is None:
            get_context_events().add_event(
                customer_id=customer_id,
                source="routine",
                event_type=event_type,
                payload=queue_payload,
            )
            return
        try:
            replies = await get_telegram_chat().relay_event(
                customer_id=customer_id,
                event_label=f"routine/{event_type}",
                payload=queue_payload,
                agent_runtime=runtime,
            )
        except Exception:
            get_context_events().add_event(
                customer_id=customer_id,
                source="routine",
                event_type=event_type,
                payload=queue_payload,
            )
            return
        if not replies:
            get_context_events().add_event(
                customer_id=customer_id,
                source="routine",
                event_type=event_type,
                payload=queue_payload,
            )
            return
        for item in replies:
            await get_telegram_client().send_message(
                chat_id=item["chat_id"],
                text=item["text"],
                parse_mode="HTML",
            )

    wake_queue_service = WakeQueueService(
        db_path=PROJECT_ROOT / ".opentulpa" / "wake_events.db",
        handler=process_wake_event,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loader = get_tulpa_loader()
        result = loader.reload()
        if result.get("errors"):
            logger.warning("Tulpa router load had errors: %s", result["errors"])
        if runtime and hasattr(runtime, "start"):
            await runtime.start()
        if scheduler_service:
            scheduler_service.start()
        if task_runner:
            await task_runner.start()
        if wake_queue_service:
            await wake_queue_service.start()
        yield
        if scheduler_service:
            scheduler_service.shutdown(wait=True)
        if task_runner:
            await task_runner.shutdown()
        if wake_queue_service:
            await wake_queue_service.shutdown()
        if runtime and hasattr(runtime, "shutdown"):
            await runtime.shutdown()

    app = FastAPI(
        title="OpenTulpa",
        description="Background-capable agent API",
        version="0.1.0",
        lifespan=lifespan,
    )
    tulpa_loader = TulpaRouterLoader(project_root=PROJECT_ROOT, mount_router=tulpa_router)
    app.include_router(tulpa_router, prefix="/tulpa")

    register_health_routes(app, get_agent_runtime=get_agent_runtime)
    register_memory_routes(app, get_memory=get_memory)
    register_file_routes(
        app,
        get_file_vault=get_file_vault,
        get_telegram_chat=get_telegram_chat,
        get_telegram_client=get_telegram_client,
        get_agent_runtime=get_agent_runtime,
        telegram_enabled=bool(settings.telegram_bot_token),
    )
    register_profile_routes(
        app,
        get_profiles=get_profiles,
        get_memory=lambda: memory_service,
    )
    register_skill_routes(
        app,
        get_skill_store=get_skill_store,
        get_memory=lambda: memory_service,
    )

    decide_approval_and_maybe_wake = register_approval_routes(
        app,
        get_approvals=get_approvals,
        get_wake_queue=get_wake_queue,
        get_agent_runtime=get_agent_runtime,
    )
    register_scheduler_routes(
        app,
        get_scheduler=get_scheduler,
        delete_file=sandbox_delete_file,
    )
    register_wake_and_search_routes(
        app,
        get_wake_queue=get_wake_queue,
        llm_model=settings.llm_model,
    )
    register_tulpa_routes(app, get_tulpa_loader=get_tulpa_loader)
    register_task_routes(app, get_tasks=get_tasks)

    if slack_service is not None:
        register_slack_routes(
            app,
            get_slack=get_slack,
            has_write_consent=has_slack_write_consent,
            grant_write_consent=grant_slack_write_consent,
        )

    register_telegram_webhook_routes(
        app,
        settings=settings,
        get_telegram_client=get_telegram_client,
        get_telegram_chat=get_telegram_chat,
        get_agent_runtime=get_agent_runtime,
        get_context_events=get_context_events,
        decide_approval_and_maybe_wake=decide_approval_and_maybe_wake,
    )

    return app
