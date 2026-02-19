"""OpenTulpa entry point."""

import os
import sys
from pathlib import Path

import uvicorn

from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime
from opentulpa.api.app import create_app
from opentulpa.context.customer_profiles import CustomerProfileService
from opentulpa.context.file_vault import FileVaultService
from opentulpa.context.service import EventContextService
from opentulpa.context.thread_rollups import ThreadRollupService
from opentulpa.core.config import get_settings
from opentulpa.integrations.slack_client import SlackClient
from opentulpa.memory.service import MemoryService
from opentulpa.scheduler.service import SchedulerService
from opentulpa.skills.service import SkillStoreService
from opentulpa.tasks.service import TaskService


async def _wake_callback(payload: dict) -> None:
    """Called when a scheduled routine fires; posts to /internal/wake."""
    import httpx

    settings = get_settings()
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"http://127.0.0.1:{settings.port}/internal/wake",
                json=payload,
                timeout=5.0,
            )
    except Exception:
        pass


def _mem0_config_openrouter(
    llm_model: str,
    embedding_model: str,
    openrouter_api_key: str | None,
    openrouter_base_url: str,
    qdrant_path: str,
    qdrant_on_disk: bool,
) -> dict:
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": f"google/{llm_model}",
                "api_key": openrouter_api_key,
                "openai_base_url": openrouter_base_url,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": embedding_model,
                "openai_base_url": openrouter_base_url,
                "api_key": openrouter_api_key,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "mem0",
                "path": qdrant_path,
                "on_disk": qdrant_on_disk,
            },
        },
    }


def main() -> None:
    settings = get_settings()
    project_root = Path(__file__).resolve().parents[2]
    openrouter_api_key = settings.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    qdrant_path = Path(settings.mem0_qdrant_path)
    if not qdrant_path.is_absolute():
        qdrant_path = project_root / qdrant_path

    memory = MemoryService(
        user_id=settings.mem0_user_id,
        config=_mem0_config_openrouter(
            settings.llm_model,
            settings.openrouter_embedding_model,
            openrouter_api_key,
            settings.openrouter_base_url,
            str(qdrant_path),
            settings.mem0_qdrant_on_disk,
        ),
    )
    scheduler = SchedulerService()
    scheduler.set_wake_callback(_wake_callback)
    context_events = EventContextService(db_path=project_root / ".opentulpa" / "context_events.db")
    customer_profiles = CustomerProfileService(
        db_path=project_root / ".opentulpa" / "customer_profiles.db"
    )
    customer_profiles.import_legacy(
        directives_db_path=project_root / ".opentulpa" / "directives.db",
        time_profiles_db_path=project_root / ".opentulpa" / "time_profiles.db",
    )
    file_vault = FileVaultService(
        root_dir=project_root / ".opentulpa" / "file_vault",
        db_path=project_root / ".opentulpa" / "file_vault.db",
    )
    thread_rollups = ThreadRollupService(db_path=project_root / ".opentulpa" / "thread_rollups.db")
    skill_store = SkillStoreService(
        db_path=project_root / ".opentulpa" / "skills.db",
        root_dir=project_root / ".opentulpa" / "skills",
    )
    skill_store.ensure_default_skill()
    task_service = TaskService(
        db_path=project_root / ".opentulpa" / "tasks.db",
        wake_callback=_wake_callback,
    )
    slack_client = SlackClient(settings.slack_bot_token) if settings.slack_bot_token else None

    agent_runtime: OpenTulpaLangGraphRuntime | None = None
    if openrouter_api_key:
        agent_runtime = OpenTulpaLangGraphRuntime(
            app_url=f"http://127.0.0.1:{settings.port}",
            openrouter_api_key=openrouter_api_key,
            model_name=settings.llm_model,
            checkpoint_db_path=settings.agent_checkpoint_db_path,
            recursion_limit=settings.agent_recursion_limit,
            context_events=context_events,
            customer_profile_service=customer_profiles,
            thread_rollup_service=thread_rollups,
            context_token_limit=settings.agent_context_token_limit,
            context_rollup_tokens=settings.agent_context_rollup_tokens,
        )
    else:
        print(
            "OPENROUTER_API_KEY is not set; starting FastAPI without AI chat backend. "
            "Set key and restart to enable full chat.",
            file=sys.stderr,
        )

    app = create_app(
        memory=memory,
        scheduler=scheduler,
        slack_client=slack_client,
        task_service=task_service,
        agent_runtime=agent_runtime,
        context_events=context_events,
        customer_profile_service=customer_profiles,
        file_vault_service=file_vault,
        skill_store_service=skill_store,
    )
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
