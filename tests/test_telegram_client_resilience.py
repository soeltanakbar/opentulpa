from __future__ import annotations

import httpx
import pytest

from opentulpa.interfaces.telegram import client as telegram_client_module
from opentulpa.interfaces.telegram.client import TelegramClient


class _AlwaysTimeoutClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        raise httpx.ReadTimeout("timeout")


class _RetryThenSuccessClient:
    attempts = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        _RetryThenSuccessClient.attempts += 1
        if _RetryThenSuccessClient.attempts < 3:
            raise httpx.TransportError("temporary transport failure")
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 123}})


class _NotModifiedEditClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        return httpx.Response(
            400,
            json={
                "ok": False,
                "error_code": 400,
                "description": "Bad Request: message is not modified",
            },
        )


@pytest.mark.asyncio
async def test_post_returns_none_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(telegram_client_module.httpx, "AsyncClient", _AlwaysTimeoutClient)
    tg = TelegramClient("dummy")
    result = await tg._post("sendMessage", {"chat_id": 1, "text": "hello"})
    assert result is None


@pytest.mark.asyncio
async def test_post_retries_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    _RetryThenSuccessClient.attempts = 0
    monkeypatch.setattr(telegram_client_module.httpx, "AsyncClient", _RetryThenSuccessClient)
    tg = TelegramClient("dummy")
    result = await tg._post("sendMessage", {"chat_id": 1, "text": "hello"})
    assert isinstance(result, dict)
    assert result.get("ok") is True
    assert _RetryThenSuccessClient.attempts == 3


@pytest.mark.asyncio
async def test_post_treats_not_modified_edit_as_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(telegram_client_module.httpx, "AsyncClient", _NotModifiedEditClient)
    tg = TelegramClient("dummy")
    result = await tg._post("editMessageText", {"chat_id": 1, "message_id": 10, "text": "..."})
    assert isinstance(result, dict)
    assert result.get("ok") is True
