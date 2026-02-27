"""Application-layer orchestration for skill APIs."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import Any

from opentulpa.application.contracts import ApplicationResult
from opentulpa.skills.service import build_skill_markdown


class SkillOrchestratorResult(ApplicationResult[dict[str, object]]):
    """Normalized route-friendly result payload."""


class SkillOrchestrator:
    """Owns skill endpoint business rules independent of FastAPI transport."""

    def __init__(
        self,
        *,
        get_skill_store: Callable[[], Any],
        get_memory: Callable[[], Any],
    ) -> None:
        self._get_skill_store = get_skill_store
        self._get_memory = get_memory

    def list_skills(
        self,
        *,
        customer_id: str,
        include_global: bool,
        include_disabled: bool,
        limit: int,
    ) -> SkillOrchestratorResult:
        store = self._get_skill_store()
        skills = store.list_skills(
            customer_id=customer_id,
            include_global=include_global,
            include_disabled=include_disabled,
            limit=limit,
        )
        return SkillOrchestratorResult(status_code=200, payload={"ok": True, "skills": skills})

    def get_skill(
        self,
        *,
        customer_id: str,
        name: str,
        include_files: bool,
        include_global: bool,
    ) -> SkillOrchestratorResult:
        safe_name = str(name or "").strip()
        if not safe_name:
            return SkillOrchestratorResult(status_code=400, payload={"detail": "name is required"})
        store = self._get_skill_store()
        skill = store.get_skill(
            customer_id=customer_id,
            name=safe_name,
            include_files=include_files,
            include_global=include_global,
        )
        if skill is None:
            return SkillOrchestratorResult(status_code=404, payload={"detail": "skill not found"})
        return SkillOrchestratorResult(status_code=200, payload={"ok": True, "skill": skill})

    def upsert_skill(
        self,
        *,
        customer_id: str,
        scope: str,
        name: str,
        description: str,
        instructions: str,
        skill_markdown: str,
        source: str,
        supporting_files: dict[str, str] | None,
    ) -> SkillOrchestratorResult:
        store = self._get_skill_store()
        safe_customer_id = str(customer_id or "").strip()
        safe_scope = str(scope or "").strip().lower()
        safe_name = str(name or "").strip()
        safe_description = str(description or "").strip()
        safe_instructions = str(instructions or "").strip()
        safe_skill_markdown = str(skill_markdown or "").strip()
        safe_source = str(source or "agent")

        if safe_scope == "user" and not safe_customer_id:
            return SkillOrchestratorResult(
                status_code=400,
                payload={"detail": "customer_id is required for user skills"},
            )
        if not safe_name:
            return SkillOrchestratorResult(status_code=400, payload={"detail": "name is required"})
        try:
            if not safe_skill_markdown:
                safe_skill_markdown = build_skill_markdown(
                    name=safe_name,
                    description=safe_description,
                    instructions=safe_instructions,
                )
            skill = store.upsert_skill(
                scope=safe_scope,
                customer_id=safe_customer_id,
                name=safe_name,
                skill_markdown=safe_skill_markdown,
                source=safe_source,
                enabled=True,
                supporting_files=supporting_files,
            )
        except Exception as exc:
            return SkillOrchestratorResult(status_code=400, payload={"detail": str(exc)})

        memory = self._get_memory()
        if memory is not None:
            with suppress(Exception):
                memory.add_text(
                    (
                        "Skill stored for this user: "
                        f"name={skill.get('name')} scope={skill.get('scope')} "
                        f"description={skill.get('description')}"
                    ),
                    user_id=safe_customer_id or "global",
                    metadata={
                        "kind": "user_skill",
                        "skill_name": skill.get("name"),
                        "scope": skill.get("scope"),
                    },
                )
        return SkillOrchestratorResult(status_code=200, payload={"ok": True, "skill": skill})

    def delete_skill(
        self,
        *,
        customer_id: str,
        scope: str,
        name: str,
    ) -> SkillOrchestratorResult:
        store = self._get_skill_store()
        safe_customer_id = str(customer_id or "").strip()
        safe_scope = str(scope or "").strip().lower()
        safe_name = str(name or "").strip()
        if not safe_name:
            return SkillOrchestratorResult(status_code=400, payload={"detail": "name is required"})
        if safe_scope == "user" and not safe_customer_id:
            return SkillOrchestratorResult(
                status_code=400,
                payload={"detail": "customer_id is required for user skills"},
            )
        try:
            deleted = store.delete_skill(scope=safe_scope, customer_id=safe_customer_id, name=safe_name)
        except Exception as exc:
            return SkillOrchestratorResult(status_code=400, payload={"detail": str(exc)})
        return SkillOrchestratorResult(
            status_code=200,
            payload={"ok": True, "deleted": bool(deleted)},
        )
