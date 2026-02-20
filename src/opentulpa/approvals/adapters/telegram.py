"""Telegram adapter for approval challenges."""

from __future__ import annotations

import json
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

    @staticmethod
    def _action_preview(approval: ApprovalRecord) -> str:
        summary = str(approval.summary or "").strip() or f"execute {approval.action_name}"
        try:
            args = json.loads(str(approval.action_args_json or "{}"))
        except Exception:
            args = {}
        if not isinstance(args, dict):
            args = {}

        action_name = str(approval.action_name or "").strip()
        preview = summary
        if action_name == "browser_use_run":
            task_text = str(args.get("task", "")).strip()
            if task_text:
                preview = f"run browser task:\n{task_text}"
        elif action_name == "tulpa_run_terminal":
            cmd = str(args.get("command", "")).strip()
            if cmd:
                preview = f"run terminal command:\n{cmd}"

        # Keep message below Telegram hard limits while avoiding aggressive truncation.
        max_chars = 2400
        if len(preview) > max_chars:
            preview = preview[: max_chars - 3].rstrip() + "..."
        return preview

    async def send_challenge(self, approval: ApprovalRecord) -> bool:
        chat_id = str(approval.origin_conversation_id or "").strip()
        if not chat_id:
            return False
        expiry = self._format_expiry(approval.expires_at)
        action_preview = self._action_preview(approval)
        text = (
            "Approval needed for external-impact action.\n\n"
            f"Action: {action_preview}\n"
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
