from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from opentulpa.api.app import create_app
from opentulpa.scheduler.models import Routine
from opentulpa.scheduler.service import SchedulerService


def _mk_client(tmp_path: Path) -> TestClient:
    scheduler = SchedulerService(db_path=tmp_path / "scheduler.db")
    scheduler.add_routine(
        Routine(
            id="rtn_auto1",
            name="Hourly Trending GIFs",
            schedule="0 * * * *",
            payload={
                "customer_id": "telegram_1",
                "cleanup_paths": ["tulpa_stuff/scripts/giphy_trending.py"],
            },
            is_cron=True,
        )
    )
    app = create_app(scheduler=scheduler)
    return TestClient(app)


def test_delete_automation_with_assets(tmp_path: Path, monkeypatch) -> None:
    deleted_paths: list[tuple[str, bool]] = []

    def _fake_delete(path: str, *, missing_ok: bool = True) -> dict[str, object]:
        deleted_paths.append((path, missing_ok))
        return {"ok": True, "deleted": True, "path": path}

    monkeypatch.setattr("opentulpa.api.app.sandbox_delete_file", _fake_delete)

    with _mk_client(tmp_path) as client:
        response = client.post(
            "/internal/scheduler/routine/delete_with_assets",
            json={
                "customer_id": "telegram_1",
                "routine_id": "rtn_auto1",
                "delete_files": True,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["ok"] is True
        assert payload["deleted_routines"] == [{"id": "rtn_auto1", "name": "Hourly Trending GIFs"}]
        assert deleted_paths == [("tulpa_stuff/scripts/giphy_trending.py", True)]

        listed = client.get("/internal/scheduler/routines", params={"customer_id": "telegram_1"})
        assert listed.status_code == 200
        assert listed.json()["routines"] == []
