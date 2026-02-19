"""Text-token fallback adapter for approval challenges."""

from __future__ import annotations

from opentulpa.approvals.models import ApprovalRecord
from opentulpa.interfaces.telegram.client import TelegramClient


class TextTokenApprovalAdapter:
    name = "text_token"
    interactive = False

    def __init__(self, *, telegram_client: TelegramClient | None = None) -> None:
        self._telegram_client = telegram_client

    async def send_challenge(self, approval: ApprovalRecord) -> bool:
        if self._telegram_client is None:
            return False
        chat_id = str(approval.origin_conversation_id or "").strip()
        if not chat_id:
            return False
        text = (
            "Approval required before I can continue.\n\n"
            f"Action: {approval.summary}\n"
            f"Risk: {approval.impact_type}\n"
            f"ID: {approval.id}\n\n"
            f"Reply with /approve {approval.id} or /deny {approval.id}."
        )
        return await self._telegram_client.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="HTML",
        )
