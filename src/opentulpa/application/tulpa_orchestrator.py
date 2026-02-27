"""Application-layer orchestration for tulpa sandbox APIs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opentulpa.application.contracts import ApplicationResult
from opentulpa.tasks.sandbox import (
    ALLOWED_TERMINAL_COMMANDS,
    ALLOWED_TERMINAL_DIRS,
    PROJECT_ROOT,
    get_tulpa_catalog,
)
from opentulpa.tasks.sandbox import read_file as sandbox_read_file
from opentulpa.tasks.sandbox import run_terminal as sandbox_run_terminal
from opentulpa.tasks.sandbox import validate_generated_file as sandbox_validate_generated_file
from opentulpa.tasks.sandbox import write_file as sandbox_write_file


class TulpaOrchestratorResult(ApplicationResult[dict[str, object]]):
    """Normalized route-friendly result payload."""


class TulpaOrchestrator:
    """Owns tulpa endpoint business rules independent of FastAPI transport."""

    def __init__(self, *, get_tulpa_loader: Callable[[], Any]) -> None:
        self._get_tulpa_loader = get_tulpa_loader

    def reload_modules(self) -> TulpaOrchestratorResult:
        return TulpaOrchestratorResult(status_code=200, payload=self._get_tulpa_loader().reload())

    def write_file(self, *, relative_path: str, content: Any | None) -> TulpaOrchestratorResult:
        if content is None:
            return TulpaOrchestratorResult(status_code=400, payload={"detail": "content is required"})
        try:
            target = sandbox_write_file(relative_path, str(content))
            validation = sandbox_validate_generated_file(relative_path)
        except Exception as exc:
            return TulpaOrchestratorResult(status_code=400, payload={"detail": str(exc)})
        return TulpaOrchestratorResult(
            status_code=200,
            payload={
                "ok": True,
                "path": str(target.relative_to(PROJECT_ROOT)),
                "validation": validation,
            },
        )

    def validate_file(self, *, relative_path: str) -> TulpaOrchestratorResult:
        try:
            result = sandbox_validate_generated_file(relative_path)
        except Exception as exc:
            return TulpaOrchestratorResult(status_code=400, payload={"detail": str(exc)})
        return TulpaOrchestratorResult(status_code=200, payload=result)

    def read_file(self, *, path: str, max_chars: int) -> TulpaOrchestratorResult:
        try:
            content = sandbox_read_file(path, max_chars=max_chars)
        except Exception as exc:
            return TulpaOrchestratorResult(status_code=400, payload={"detail": str(exc)})
        return TulpaOrchestratorResult(
            status_code=200,
            payload={"ok": True, "path": path, "content": content},
        )

    def run_terminal(
        self,
        *,
        command: str,
        working_dir_key: str,
        timeout_seconds: int,
    ) -> TulpaOrchestratorResult:
        if not command:
            return TulpaOrchestratorResult(status_code=400, payload={"detail": "command is required"})
        try:
            payload = sandbox_run_terminal(
                command=command,
                working_dir=working_dir_key,
                timeout_seconds=timeout_seconds,
            )
            return TulpaOrchestratorResult(status_code=200, payload=payload)
        except PermissionError as exc:
            return TulpaOrchestratorResult(
                status_code=403,
                payload={"detail": str(exc), "allowed_commands": sorted(ALLOWED_TERMINAL_COMMANDS)},
            )
        except TimeoutError as exc:
            return TulpaOrchestratorResult(status_code=408, payload={"detail": str(exc)})
        except ValueError as exc:
            return TulpaOrchestratorResult(
                status_code=400,
                payload={
                    "detail": str(exc),
                    "allowed_working_dirs": sorted(ALLOWED_TERMINAL_DIRS.keys()),
                },
            )
        except RuntimeError as exc:
            return TulpaOrchestratorResult(status_code=500, payload={"detail": str(exc)})

    def catalog(self) -> TulpaOrchestratorResult:
        return TulpaOrchestratorResult(status_code=200, payload={"ok": True, "catalog": get_tulpa_catalog()})
