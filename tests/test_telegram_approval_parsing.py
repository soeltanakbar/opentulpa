from __future__ import annotations

from opentulpa.api.routes.telegram_webhook import (
    _parse_approval_callback_data,
    _parse_text_token_decision,
)


def test_parse_approval_callback_data_accepts_approve() -> None:
    parsed = _parse_approval_callback_data("approval:apr_abc123:approve")
    assert parsed == ("apr_abc123", "approve")


def test_parse_text_token_decision_accepts_deny() -> None:
    parsed = _parse_text_token_decision("/deny apr_abc123")
    assert parsed == ("apr_abc123", "deny")


def test_parse_approval_callback_data_rejects_approve_always() -> None:
    parsed = _parse_approval_callback_data("approval:apr_abc123:approve_always")
    assert parsed is None


def test_parse_text_token_decision_rejects_approve_forever() -> None:
    parsed = _parse_text_token_decision("/approve_forever apr_abc123")
    assert parsed is None
