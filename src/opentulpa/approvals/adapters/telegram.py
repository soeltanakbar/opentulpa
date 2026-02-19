"""Telegram adapter for approval challenges."""

from __future__ import annotations

from datetime import datetime

from opentulpa.approvals.models import ApprovalRecord
from opentulpa.interfaces.telegram.client import TelegramClient


class TelegramApprovalAdapter:
    name = "telegram"
    interactive = True

    def __init__(self, *, client: TelegramClient) -> None:
        self._client = client

    @staticmethod
    def _format_expiry(expires_at: str) -> str:
        try:
            dt = datetime.fromisoformat(expires_at)
            return dt.strftime("%H:%M:%S")
        except Exception:
            return "soon"

    async def send_challenge(self, approval: ApprovalRecord) -> bool:
        chat_id = str(approval.origin_conversation_id or "").strip()
        if not chat_id:
            return False
        expiry = self._format_expiry(approval.expires_at)
        text = (
            "Approval needed for external-impact action.\n\n"
            f"Action: {approval.summary}\n"
            f"Destination: {approval.recipient_scope}\n"
            f"Risk: {approval.impact_type}\n"
            f"Expires: {expiry}\n\n"
            f"ID: {approval.id}"
        )
        reply_markup = {
            "inline_keyboard": [
                [
                    {"text": "Approve", "callback_data": f"approval:{approval.id}:approve"},
                    {"text": "Deny", "callback_data": f"approval:{approval.id}:deny"},
                ]
            ]
        }
        return await self._client.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="HTML",
            reply_markup=reply_markup,
        )
