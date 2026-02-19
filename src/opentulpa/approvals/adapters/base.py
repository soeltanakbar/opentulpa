"""Interface adapter contracts for approval challenges."""

from __future__ import annotations

from typing import Protocol

from opentulpa.approvals.models import ApprovalRecord


class ApprovalAdapter(Protocol):
    name: str
    interactive: bool

    async def send_challenge(self, approval: ApprovalRecord) -> bool:
        """Return True if challenge was delivered on this interface."""
