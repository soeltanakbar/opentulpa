"""Telegram API client primitives."""

from __future__ import annotations

import logging
import mimetypes
from typing import Any

import httpx

from opentulpa.interfaces.telegram.formatter import prepare_text_and_mode

logger = logging.getLogger(__name__)


class TelegramClient:
    """Thin async client around Telegram Bot API endpoints used by OpenTulpa."""

    def __init__(self, bot_token: str) -> None:
        self.bot_token = str(bot_token or "").strip()

    async def _post(self, method: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        url = f"https://api.telegram.org/bot{self.bot_token}/{method}"
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=payload, timeout=15.0)
            if not r.is_success:
                logger.warning(
                    "Telegram API %s HTTP %s: %s",
                    method,
                    r.status_code,
                    (r.text or "")[:400],
                )
                return None
            try:
                data = r.json()
            except Exception:
                logger.warning("Telegram API %s returned non-JSON body", method)
                return None
            if isinstance(data, dict) and data.get("ok") is True:
                return data
            logger.warning("Telegram API %s returned error payload: %s", method, str(data)[:400])
            return None

    async def send_message(
        self,
        *,
        chat_id: int | str,
        text: str,
        parse_mode: str | None = "HTML",
    ) -> bool:
        final_text, final_mode = prepare_text_and_mode(text, parse_mode)
        payload: dict[str, Any] = {"chat_id": chat_id, "text": final_text}
        if final_mode:
            payload["parse_mode"] = final_mode
        data = await self._post("sendMessage", payload)
        return bool(data)

    async def upsert_stream_message(
        self,
        *,
        chat_id: int | str,
        text: str,
        message_id: int | None = None,
        parse_mode: str | None = None,
    ) -> int | None:
        final_text, final_mode = prepare_text_and_mode(text, parse_mode)
        if not final_text:
            return message_id

        if message_id is None:
            payload: dict[str, Any] = {"chat_id": chat_id, "text": final_text}
            if final_mode:
                payload["parse_mode"] = final_mode
            data = await self._post("sendMessage", payload)
            if not data:
                return None
            result = data.get("result") if isinstance(data, dict) else None
            if isinstance(result, dict):
                rid = result.get("message_id")
                if isinstance(rid, int):
                    return rid
            return None

        payload = {"chat_id": chat_id, "message_id": message_id, "text": final_text}
        if final_mode:
            payload["parse_mode"] = final_mode
        data = await self._post("editMessageText", payload)
        if data:
            return message_id
        fallback_payload: dict[str, Any] = {"chat_id": chat_id, "text": final_text}
        if final_mode:
            fallback_payload["parse_mode"] = final_mode
        fallback_data = await self._post("sendMessage", fallback_payload)
        if not fallback_data:
            return message_id
        result = fallback_data.get("result") if isinstance(fallback_data, dict) else None
        if isinstance(result, dict):
            rid = result.get("message_id")
            if isinstance(rid, int):
                return rid
        return message_id

    async def download_file(self, *, file_id: str) -> dict[str, Any] | None:
        info = await self._post("getFile", {"file_id": file_id})
        if not info:
            return None
        result = info.get("result") if isinstance(info, dict) else None
        if not isinstance(result, dict):
            return None
        file_path = str(result.get("file_path", "")).strip()
        if not file_path:
            return None
        file_size = result.get("file_size")
        guessed_mime, _ = mimetypes.guess_type(file_path)
        url = f"https://api.telegram.org/file/bot{self.bot_token}/{file_path}"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=45.0)
        except Exception:
            return None
        if not resp.is_success:
            return None
        return {
            "file_path": file_path,
            "file_size": int(file_size) if isinstance(file_size, int) else len(resp.content),
            "mime_type": guessed_mime,
            "raw_bytes": resp.content,
        }

    async def send_file(
        self,
        *,
        chat_id: int | str,
        filename: str,
        raw_bytes: bytes,
        kind: str = "document",
        mime_type: str | None = None,
        caption: str | None = None,
        parse_mode: str | None = None,
    ) -> bool:
        safe_name = str(filename or "file.bin").strip() or "file.bin"
        safe_kind = str(kind or "").strip().lower()
        should_send_photo = safe_kind == "photo" and str(mime_type or "").startswith("image/")
        method = "sendPhoto" if should_send_photo else "sendDocument"
        media_field = "photo" if should_send_photo else "document"

        payload: dict[str, Any] = {"chat_id": str(chat_id)}
        if caption:
            final_caption, final_mode = prepare_text_and_mode(caption, parse_mode)
            if final_caption:
                payload["caption"] = final_caption
            if final_mode:
                payload["parse_mode"] = final_mode

        files = {media_field: (safe_name, raw_bytes, mime_type or "application/octet-stream")}
        url = f"https://api.telegram.org/bot{self.bot_token}/{method}"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, data=payload, files=files, timeout=60.0)
        except Exception:
            return False
        if not resp.is_success:
            logger.warning(
                "Telegram API %s HTTP %s: %s",
                method,
                resp.status_code,
                (resp.text or "")[:400],
            )
            return False
        try:
            data = resp.json()
        except Exception:
            logger.warning("Telegram API %s returned non-JSON body", method)
            return False
        ok = bool(isinstance(data, dict) and data.get("ok") is True)
        if not ok:
            logger.warning("Telegram API %s returned error payload: %s", method, str(data)[:400])
        return ok


def parse_telegram_update(body: dict) -> tuple[int | None, int | None, str | None]:
    """Extract (chat_id, user_id, text) from a Telegram update."""
    try:
        message = body.get("message") or body.get("edited_message")
        if not message:
            return None, None, None
        chat_id = message.get("chat", {}).get("id")
        user_id = message.get("from", {}).get("id")
        text = (message.get("text") or message.get("caption") or "").strip()
        return chat_id, user_id, text
    except Exception:
        return None, None, None
