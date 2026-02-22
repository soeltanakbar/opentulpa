from __future__ import annotations

from types import SimpleNamespace

from opentulpa.agent.graph_builder import (
    _compute_claim_check_retry_limit,
    _compute_empty_output_retry_limit,
)


def test_claim_check_retry_limit_uses_recursion_budget() -> None:
    runtime = SimpleNamespace(recursion_limit=30)
    assert _compute_claim_check_retry_limit(runtime) == 24


def test_claim_check_retry_limit_has_floor() -> None:
    runtime = SimpleNamespace(recursion_limit=6)
    assert _compute_claim_check_retry_limit(runtime) == 3


def test_claim_check_retry_limit_handles_missing_runtime_attr() -> None:
    runtime = SimpleNamespace()
    assert _compute_claim_check_retry_limit(runtime) == 24


def test_empty_output_retry_limit_is_capped_small() -> None:
    runtime = SimpleNamespace(recursion_limit=80)
    assert _compute_empty_output_retry_limit(runtime) == 2


def test_empty_output_retry_limit_respects_floor() -> None:
    runtime = SimpleNamespace(recursion_limit=6)
    assert _compute_empty_output_retry_limit(runtime) == 2
