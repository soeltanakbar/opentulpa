"""Shared graph retry-limit helpers."""

from __future__ import annotations

from typing import Any


def compute_claim_check_retry_limit(runtime: Any) -> int:
    """Derive retry budget from graph recursion limit."""
    try:
        recursion_limit = int(getattr(runtime, "recursion_limit", 30))
    except Exception:
        recursion_limit = 30
    return max(3, min(24, recursion_limit - 6))


def compute_empty_output_retry_limit(runtime: Any) -> int:
    """Empty assistant outputs should self-repair quickly then exit."""
    return min(2, compute_claim_check_retry_limit(runtime))
