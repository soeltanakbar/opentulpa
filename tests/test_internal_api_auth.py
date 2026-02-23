from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from opentulpa.api.app import create_app
from opentulpa.core.config import get_settings
from opentulpa.skills.service import SkillStoreService


def _mk_client(tmp_path: Path, *, client_host: str = "127.0.0.1") -> TestClient:
    store = SkillStoreService(
        db_path=tmp_path / "skills.db",
        root_dir=tmp_path / "skills",
    )
    app = create_app(skill_store_service=store)
    return TestClient(app, client=(client_host, 50000))


def test_internal_routes_allow_server_local_traffic(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    get_settings.cache_clear()
    with _mk_client(tmp_path, client_host="127.0.0.1") as client:
        no_header = client.post("/internal/skills/list", json={"customer_id": "telegram_1"})
        assert no_header.status_code == 200
    get_settings.cache_clear()


def test_internal_routes_blocked_from_public_clients(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    get_settings.cache_clear()
    with _mk_client(tmp_path, client_host="8.8.8.8") as client:
        response = client.post("/internal/skills/list", json={"customer_id": "telegram_1"})
        assert response.status_code == 403

        healthz = client.get("/healthz")
        assert healthz.status_code == 200

        agent_healthz = client.get("/agent/healthz")
        assert agent_healthz.status_code == 200
    get_settings.cache_clear()


def test_webhook_route_public_with_telegram_auth(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "tg-secret")
    get_settings.cache_clear()
    with _mk_client(tmp_path, client_host="8.8.8.8") as client:
        response = client.post(
            "/webhook/telegram",
            json={},
            headers={"x-telegram-bot-api-secret-token": "tg-secret"},
        )
        assert response.status_code == 200
    get_settings.cache_clear()
