"""Telegram attachment extraction and ingest pipeline."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from opentulpa.context.file_vault import FileVaultService
from opentulpa.interfaces.telegram.client import TelegramClient
from opentulpa.interfaces.telegram.models import TelegramAttachment


def extract_attachments(message: dict[str, Any]) -> list[TelegramAttachment]:
    attachments: list[TelegramAttachment] = []

    document = message.get("document")
    if isinstance(document, dict):
        fid = str(document.get("file_id", "")).strip()
        if fid:
            attachments.append(
                TelegramAttachment(
                    kind="document",
                    file_id=fid,
                    filename=str(document.get("file_name", "")).strip() or None,
                    mime_type=str(document.get("mime_type", "")).strip() or None,
                )
            )

    photos = message.get("photo")
    if isinstance(photos, list) and photos:
        chosen: dict[str, Any] | None = None
        for item in photos:
            if not isinstance(item, dict):
                continue
            if chosen is None or int(item.get("file_size") or 0) >= int(chosen.get("file_size") or 0):
                chosen = item
        if chosen:
            fid = str(chosen.get("file_id", "")).strip()
            if fid:
                unique = str(chosen.get("file_unique_id", "")).strip() or "photo"
                attachments.append(
                    TelegramAttachment(
                        kind="photo",
                        file_id=fid,
                        filename=f"{unique}.jpg",
                        mime_type="image/jpeg",
                    )
                )

    for key in ("video", "audio", "voice"):
        item = message.get(key)
        if not isinstance(item, dict):
            continue
        fid = str(item.get("file_id", "")).strip()
        if not fid:
            continue
        unique = str(item.get("file_unique_id", "")).strip() or key
        ext = {
            "video": ".mp4",
            "audio": ".mp3",
            "voice": ".ogg",
        }.get(key, "")
        filename = str(item.get("file_name", "")).strip() or f"{unique}{ext}"
        attachments.append(
            TelegramAttachment(
                kind=key,
                file_id=fid,
                filename=filename,
                mime_type=str(item.get("mime_type", "")).strip() or None,
            )
        )
    return attachments


def build_uploaded_files_context(records: list[dict[str, Any]]) -> str:
    if not records:
        return ""
    lines = [
        "Uploaded files attached to this message were already ingested and indexed:",
    ]
    for rec in records:
        lines.append(
            "- id={id} name={name} kind={kind} created_at={created_at} summary={summary}".format(
                id=str(rec.get("id", "")).strip(),
                name=str(rec.get("original_filename", "")).strip(),
                kind=str(rec.get("kind", "")).strip(),
                created_at=str(rec.get("created_at", "")).strip(),
                summary=str(rec.get("summary", "")).strip()[:700],
            )
        )
    return "\n".join(lines)


async def ingest_attachments(
    *,
    attachments: list[TelegramAttachment],
    bot_token: str,
    file_vault: FileVaultService,
    memory: Any | None,
    agent_runtime: Any | None,
    customer_id: str,
    chat_id: int,
    caption: str | None,
) -> list[dict[str, Any]]:
    ingested: list[dict[str, Any]] = []
    client = TelegramClient(bot_token)
    for attachment in attachments:
        downloaded = await client.download_file(file_id=attachment.file_id)
        if not downloaded:
            continue
        raw_bytes = downloaded.get("raw_bytes")
        if not isinstance(raw_bytes, (bytes, bytearray)) or not raw_bytes:
            continue
        file_path_name = str(downloaded.get("file_path", "")).split("/")[-1].strip()
        record = file_vault.ingest_file(
            customer_id=customer_id,
            chat_id=chat_id,
            kind=attachment.kind,
            telegram_file_id=attachment.file_id,
            original_filename=attachment.filename or file_path_name or f"{attachment.kind}.bin",
            mime_type=attachment.mime_type or str(downloaded.get("mime_type", "")).strip() or None,
            caption=caption,
            raw_bytes=bytes(raw_bytes),
        )
        if (
            attachment.kind == "voice"
            and agent_runtime is not None
            and hasattr(agent_runtime, "transcribe_audio_blob")
        ):
            with suppress(Exception):
                transcript = await agent_runtime.transcribe_audio_blob(
                    filename=attachment.filename or file_path_name or f"{attachment.kind}.ogg",
                    mime_type=attachment.mime_type
                    or str(downloaded.get("mime_type", "")).strip()
                    or None,
                    kind=attachment.kind,
                    raw_bytes=bytes(raw_bytes),
                )
                if transcript:
                    record = {**record, "voice_transcript": str(transcript).strip()[:4000]}
        if agent_runtime is not None and hasattr(agent_runtime, "summarize_uploaded_blob"):
            if attachment.kind == "voice":
                ingested.append(record)
                if memory is not None:
                    with suppress(Exception):
                        memory.add_text(
                            (
                                "User sent voice message: "
                                f"id={record.get('id')} transcript={str(record.get('voice_transcript', ''))[:1200]}"
                            ),
                            user_id=customer_id,
                            metadata={
                                "kind": "uploaded_voice_message",
                                "file_id": record.get("id"),
                                "file_kind": record.get("kind"),
                            },
                        )
                continue
            with suppress(Exception):
                ai_summary = await agent_runtime.summarize_uploaded_blob(
                    filename=str(record.get("original_filename", "")).strip() or None,
                    mime_type=str(record.get("mime_type", "")).strip() or None,
                    kind=str(record.get("kind", "")).strip() or None,
                    raw_bytes=bytes(raw_bytes),
                    caption=caption,
                )
                if ai_summary:
                    updated = file_vault.set_ai_summary(customer_id, str(record.get("id", "")), ai_summary)
                    if isinstance(updated, dict):
                        record = updated
        ingested.append(record)
        if memory is not None:
            with suppress(Exception):
                memory.add_text(
                    (
                        "User uploaded file stored in vault: "
                        f"id={record.get('id')} name={record.get('original_filename')} "
                        f"kind={record.get('kind')} summary={record.get('summary', '')[:1200]}"
                    ),
                    user_id=customer_id,
                    metadata={
                        "kind": "uploaded_file",
                        "file_id": record.get("id"),
                        "file_kind": record.get("kind"),
                    },
                )
    return ingested
