from __future__ import annotations

from opentulpa.api.routes.telegram_webhook import (
    _parse_approval_callback_data,
)


def test_parse_approval_callback_data_accepts_approve() -> None:
    parsed = _parse_approval_callback_data("approval:apr_abc123:approve")
    assert parsed == ("apr_abc123", "approve")

def test_parse_approval_callback_data_rejects_approve_always() -> None:
    parsed = _parse_approval_callback_data("approval:apr_abc123:approve_always")
    assert parsed is None
