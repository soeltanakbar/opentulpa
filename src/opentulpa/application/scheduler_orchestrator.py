"""Application-layer orchestration for scheduler routine APIs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opentulpa.application.contracts import ApplicationResult
from opentulpa.core.ids import new_short_id
from opentulpa.scheduler.models import Routine
from opentulpa.scheduler.service import SchedulerService


class SchedulerOrchestratorResult(ApplicationResult[dict[str, object]]):
    """Normalized route-friendly result payload."""


class SchedulerOrchestrator:
    """Owns routine CRUD business rules independent of FastAPI transport."""

    def __init__(
        self,
        *,
        get_scheduler: Callable[[], SchedulerService],
        delete_file: Callable[..., dict[str, object]],
    ) -> None:
        self._get_scheduler = get_scheduler
        self._delete_file = delete_file

    @staticmethod
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

    @classmethod
    def _collect_routine_cleanup_paths(cls, payload: dict[str, object]) -> list[str]:
        if not isinstance(payload, dict):
            return []
        candidates: list[str] = []
        list_keys = ("cleanup_paths", "script_paths", "file_paths")
        scalar_keys = ("cleanup_path", "script_path", "file_path")
        for key in list_keys:
            candidates.extend(cls._normalize_cleanup_paths(payload.get(key)))
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

    @staticmethod
    def _serialize_routine(routine: Routine) -> dict[str, object]:
        return {
            "id": routine.id,
            "name": routine.name,
            "schedule": routine.schedule,
            "enabled": routine.enabled,
            "is_cron": routine.is_cron,
        }

    def create_routine(
        self,
        *,
        routine_id: str,
        name: str,
        schedule: str,
        payload: dict[str, object],
        enabled: bool,
        is_cron: bool,
    ) -> SchedulerOrchestratorResult:
        sched = self._get_scheduler()
        rid = str(routine_id or "").strip() or new_short_id("rtn")
        routine = Routine(
            id=rid,
            name=name,
            schedule=schedule,
            payload=payload if isinstance(payload, dict) else {},
            enabled=bool(enabled),
            is_cron=bool(is_cron),
        )
        sched.add_routine(routine)
        return SchedulerOrchestratorResult(status_code=200, payload={"ok": True, "id": rid})

    def list_routines(self, *, customer_id: str | None = None) -> SchedulerOrchestratorResult:
        sched = self._get_scheduler()
        routines = sched.list_routines()
        cid = str(customer_id or "").strip()
        if cid:
            routines = [
                routine
                for routine in routines
                if str((routine.payload or {}).get("customer_id", "")).strip() == cid
            ]
        return SchedulerOrchestratorResult(
            status_code=200,
            payload={"routines": [self._serialize_routine(routine) for routine in routines]},
        )

    def remove_routine(
        self,
        *,
        routine_id: str,
        customer_id: str | None = None,
    ) -> SchedulerOrchestratorResult:
        sched = self._get_scheduler()
        cid = str(customer_id or "").strip()
        if cid:
            routine = sched.get_routine(routine_id)
            if routine is None:
                return SchedulerOrchestratorResult(status_code=200, payload={"ok": False})
            owner = str((routine.payload or {}).get("customer_id", "")).strip()
            if owner != cid:
                return SchedulerOrchestratorResult(
                    status_code=403,
                    payload={"detail": "routine does not belong to this customer_id"},
                )
        ok = sched.remove_routine(routine_id)
        return SchedulerOrchestratorResult(status_code=200, payload={"ok": ok})

    def remove_routine_with_assets(
        self,
        *,
        customer_id: str,
        routine_id: str,
        name: str,
        remove_all_matches: bool,
        delete_files: bool,
        cleanup_paths: list[str] | None = None,
    ) -> SchedulerOrchestratorResult:
        sched = self._get_scheduler()
        cid = str(customer_id).strip()
        if not cid:
            return SchedulerOrchestratorResult(
                status_code=400,
                payload={"detail": "customer_id is required"},
            )

        rid = str(routine_id or "").strip()
        routine_name = str(name or "").strip()
        extra_cleanup_paths = self._normalize_cleanup_paths(cleanup_paths)

        routines = [
            routine
            for routine in sched.list_routines()
            if str((routine.payload or {}).get("customer_id", "")).strip() == cid
        ]
        if rid:
            matched = [routine for routine in routines if routine.id == rid]
        elif routine_name:
            name_cf = routine_name.casefold()
            matched = [routine for routine in routines if routine.name.strip().casefold() == name_cf]
        else:
            return SchedulerOrchestratorResult(
                status_code=400,
                payload={"detail": "routine_id or name is required"},
            )

        if not matched:
            return SchedulerOrchestratorResult(
                status_code=404,
                payload={"detail": "routine not found"},
            )
        if len(matched) > 1 and not bool(remove_all_matches):
            return SchedulerOrchestratorResult(
                status_code=400,
                payload={
                    "detail": "multiple routines matched; set remove_all_matches=true",
                    "matched_routines": [
                        {
                            "id": routine.id,
                            "name": routine.name,
                            "schedule": routine.schedule,
                        }
                        for routine in matched
                    ],
                },
            )

        deleted_routines: list[dict[str, object]] = []
        failed_routines: list[dict[str, object]] = []
        deleted_files: list[dict[str, object]] = []
        failed_files: list[dict[str, object]] = []

        for routine in matched:
            ok = sched.remove_routine(routine.id)
            if not ok:
                failed_routines.append({"id": routine.id, "name": routine.name, "error": "not found"})
                continue
            deleted_routines.append({"id": routine.id, "name": routine.name})
            if not bool(delete_files):
                continue

            cleanup = self._collect_routine_cleanup_paths(routine.payload or {})
            cleanup.extend(extra_cleanup_paths)
            unique_cleanup_paths = self._normalize_cleanup_paths(cleanup)

            for relative_path in unique_cleanup_paths:
                try:
                    result = self._delete_file(relative_path, missing_ok=True)
                    deleted_files.append(
                        {
                            "path": str(result.get("path", relative_path)),
                            "deleted": bool(result.get("deleted", False)),
                            "missing": bool(result.get("missing", False)),
                        }
                    )
                except Exception as exc:
                    failed_files.append({"path": relative_path, "error": str(exc)})

        return SchedulerOrchestratorResult(
            status_code=200,
            payload={
                "ok": len(deleted_routines) > 0 and len(failed_routines) == 0,
                "deleted_routines": deleted_routines,
                "failed_routines": failed_routines,
                "deleted_files": deleted_files,
                "failed_files": failed_files,
            },
        )
