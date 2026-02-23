from __future__ import annotations

import os
from types import SimpleNamespace

from opentulpa import __main__ as entry


def test_resolve_public_base_url_prefers_explicit_public_base(monkeypatch) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://example.com/")
    monkeypatch.setenv("RAILWAY_PUBLIC_DOMAIN", "ignored.up.railway.app")
    assert entry._resolve_public_base_url() == "https://example.com"


def test_resolve_public_base_url_falls_back_to_railway_domain(monkeypatch) -> None:
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.setenv("RAILWAY_PUBLIC_DOMAIN", "my-app.up.railway.app")
    assert entry._resolve_public_base_url() == "https://my-app.up.railway.app"


def test_ensure_telegram_webhook_secret_uses_existing(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "env-secret")
    settings = SimpleNamespace(telegram_webhook_secret="settings-secret")
    assert entry._ensure_telegram_webhook_secret(settings) == "settings-secret"


def test_ensure_telegram_webhook_secret_generates_when_missing(monkeypatch) -> None:
    monkeypatch.delenv("TELEGRAM_WEBHOOK_SECRET", raising=False)
    settings = SimpleNamespace(telegram_webhook_secret=None)
    generated = entry._ensure_telegram_webhook_secret(settings)
    assert generated
    assert os.environ.get("TELEGRAM_WEBHOOK_SECRET") == generated
