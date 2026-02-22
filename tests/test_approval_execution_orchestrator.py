from __future__ import annotations

from opentulpa.application.approval_execution import _extract_execution_error_text


def test_extract_execution_error_text_top_level_error() -> None:
    payload = {"ok": False, "error": "approval_not_found"}
    assert _extract_execution_error_text(payload) == "approval_not_found"


def test_extract_execution_error_text_nested_error() -> None:
    payload = {
        "ok": True,
        "approval_id": "apr_123",
        "status": "executed",
        "result": {"ok": False, "error": "terminal failed: working_dir invalid"},
    }
    assert _extract_execution_error_text(payload) == "terminal failed: working_dir invalid"


def test_extract_execution_error_text_nested_stderr_when_ok_false() -> None:
    payload = {
        "ok": True,
        "result": {
            "ok": False,
            "returncode": 2,
            "stderr": "python3: can't open file",
        },
    }
    assert _extract_execution_error_text(payload) == "python3: can't open file"
