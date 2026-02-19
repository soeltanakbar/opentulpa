"""Approval challenge interface adapters."""

from opentulpa.approvals.adapters.telegram import TelegramApprovalAdapter
from opentulpa.approvals.adapters.text_token import TextTokenApprovalAdapter

__all__ = ["TelegramApprovalAdapter", "TextTokenApprovalAdapter"]
