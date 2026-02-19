"""Uploaded file extraction and analysis helpers for the runtime."""

from __future__ import annotations

import base64
import mimetypes
from io import BytesIO
from typing import Any
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

import httpx

from opentulpa.agent.lc_messages import HumanMessage, SystemMessage
from opentulpa.agent.utils import content_to_text as _content_to_text


def extract_docx_text(raw_bytes: bytes) -> str:
    try:
        with ZipFile(BytesIO(raw_bytes)) as zf:
            xml_bytes = zf.read("word/document.xml")
    except (BadZipFile, KeyError) as exc:
        raise ValueError("DOCX parsing failed") from exc
    root = ElementTree.fromstring(xml_bytes)
    out: list[str] = []
    for node in root.iter():
        if node.tag.endswith("}t") and node.text:
            out.append(node.text)
    return " ".join(out).strip()


def extract_pdf_text(raw_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError("PDF parser unavailable. Install pypdf.") from exc
    try:
        reader = PdfReader(BytesIO(raw_bytes))
        parts: list[str] = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts).strip()
    except Exception as exc:
        raise ValueError(f"PDF parsing failed: {exc}") from exc


def extract_uploaded_text(
    *,
    raw_bytes: bytes,
    filename: str | None,
    mime_type: str | None,
    max_chars: int = 140000,
) -> str:
    name = str(filename or "").lower()
    mime = str(mime_type or "").lower()
    text = ""
    try:
        if mime.startswith("text/") or any(
            name.endswith(ext)
            for ext in (".txt", ".md", ".csv", ".tsv", ".json", ".yaml", ".yml", ".log")
        ):
            text = raw_bytes.decode("utf-8", errors="replace")
        elif mime == "application/pdf" or name.endswith(".pdf"):
            text = extract_pdf_text(raw_bytes)
        elif (
            mime
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            or name.endswith(".docx")
        ):
            text = extract_docx_text(raw_bytes)
    except Exception:
        text = ""
    return str(text or "").strip()[:max_chars]


def _infer_audio_format(*, filename: str | None, mime_type: str | None) -> str:
    safe_name = str(filename or "").lower().strip()
    safe_mime = str(mime_type or "").lower().split(";", 1)[0].strip()
    ext = ""
    if "." in safe_name:
        ext = safe_name.rsplit(".", 1)[-1].strip()

    ext_map = {
        "wav": "wav",
        "mp3": "mp3",
        "aiff": "aiff",
        "aac": "aac",
        "ogg": "ogg",
        "oga": "ogg",
        "flac": "flac",
        "m4a": "m4a",
    }
    if ext in ext_map:
        return ext_map[ext]

    mime_map = {
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/aiff": "aiff",
        "audio/aac": "aac",
        "audio/ogg": "ogg",
        "audio/flac": "flac",
        "audio/mp4": "m4a",
        "audio/m4a": "m4a",
    }
    return mime_map.get(safe_mime, "ogg")


async def transcribe_audio_blob(
    runtime: Any,
    *,
    filename: str | None,
    mime_type: str | None,
    kind: str | None,
    raw_bytes: bytes,
) -> str:
    """Transcribe short uploaded audio/voice files via OpenRouter input_audio."""
    safe_kind = str(kind or "").strip().lower()
    safe_mime = str(mime_type or "").lower().split(";", 1)[0].strip()
    content_bytes = bytes(raw_bytes or b"")
    if not content_bytes:
        return ""
    if safe_kind not in {"voice", "audio"} and not safe_mime.startswith("audio/"):
        return ""
    if len(content_bytes) > 12_000_000:
        return ""

    api_key = str(getattr(runtime, "openrouter_api_key", "") or "").strip()
    if not api_key:
        return ""
    base_url = (
        str(getattr(runtime, "openrouter_base_url", "") or "").strip().rstrip("/")
        or "https://openrouter.ai/api/v1"
    )
    model_name = str(getattr(runtime, "model_name", "") or "").strip()
    if not model_name:
        return ""

    audio_format = _infer_audio_format(filename=filename, mime_type=safe_mime)
    b64_audio = base64.b64encode(content_bytes).decode("ascii")
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Transcribe this audio message accurately. "
                            "Return plain text only, no commentary."
                        ),
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": b64_audio,
                            "format": audio_format,
                        },
                    },
                ],
            }
        ],
        "temperature": 0,
    }

    try:
        async with httpx.AsyncClient(timeout=75.0) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
        if resp.status_code >= 400:
            return ""
        data = resp.json()
        choices = data.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return ""
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        transcript = _content_to_text(message.get("content", "")).strip()
        return transcript[:4000]
    except Exception:
        return ""


async def summarize_uploaded_blob(
    runtime: Any,
    *,
    filename: str | None,
    mime_type: str | None,
    kind: str | None,
    raw_bytes: bytes,
    caption: str | None = None,
    question: str | None = None,
) -> str:
    safe_filename = str(filename or "file.bin").strip() or "file.bin"
    safe_mime = str(mime_type or "").strip().lower()
    if not safe_mime:
        guessed, _ = mimetypes.guess_type(safe_filename)
        safe_mime = str(guessed or "").strip().lower()
    safe_kind = str(kind or "file").strip() or "file"
    q = str(question or "").strip()
    caption_text = str(caption or "").strip()
    content_bytes = bytes(raw_bytes or b"")
    if not content_bytes:
        return f"Uploaded {safe_kind} file '{safe_filename}' was empty."

    # Gemini/OpenRouter can handle image input; keep payload bounded to avoid excessive prompt size.
    if safe_mime.startswith("image/") and len(content_bytes) <= 2_000_000:
        try:
            b64 = base64.b64encode(content_bytes).decode("ascii")
            data_url = f"data:{safe_mime};base64,{b64}"
            prompt_text = (
                "Analyze this uploaded image and summarize key information. "
                "Extract visible text, tables, IDs, dates, totals, names, and action items if present. "
                "Keep the summary concise and retrieval-friendly."
            )
            if q:
                prompt_text += f"\nUser question about this file: {q}"
            if caption_text:
                prompt_text += f"\nUser caption: {caption_text[:500]}"
            response = await runtime._model.ainvoke(
                [
                    SystemMessage(content="You analyze uploaded user files accurately."),
                    HumanMessage(
                        content=[
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ]
                    ),
                ]
            )
            vision_summary = _content_to_text(getattr(response, "content", "")).strip()
            if vision_summary:
                return vision_summary[:6000]
        except Exception:
            pass

    extracted = extract_uploaded_text(
        raw_bytes=content_bytes,
        filename=safe_filename,
        mime_type=safe_mime,
        max_chars=140000,
    )
    if extracted:
        prompt = (
            "Summarize this uploaded file for future retrieval. Include key facts, entities, "
            "dates, amounts, and concise keywords."
        )
        if q:
            prompt = (
                "Answer the user's question using only this uploaded file content. "
                "If uncertain, say what is missing."
            )
        response = await runtime._model.ainvoke(
            [
                SystemMessage(content="You analyze uploaded file content accurately and concisely."),
                HumanMessage(
                    content=(
                        f"filename={safe_filename}\n"
                        f"mime_type={safe_mime or 'unknown'}\n"
                        f"kind={safe_kind}\n"
                        f"caption={caption_text[:500]}\n"
                        f"question={q[:800]}\n\n"
                        f"{prompt}\n\n"
                        "File content:\n"
                        f"{extracted}"
                    )
                ),
            ]
        )
        text_summary = _content_to_text(getattr(response, "content", "")).strip()
        if text_summary:
            return text_summary[:6000]

    return (
        f"Uploaded {safe_kind} file '{safe_filename}' "
        f"(mime={safe_mime or 'unknown'}, size_bytes={len(content_bytes)}). "
        "No extractable text was available."
    )


async def analyze_uploaded_file(
    runtime: Any,
    *,
    record: dict[str, Any],
    raw_bytes: bytes,
    question: str | None = None,
) -> dict[str, Any]:
    analysis = await summarize_uploaded_blob(
        runtime,
        filename=str(record.get("original_filename", "")).strip() or None,
        mime_type=str(record.get("mime_type", "")).strip() or None,
        kind=str(record.get("kind", "")).strip() or None,
        raw_bytes=raw_bytes,
        caption=str(record.get("caption", "")).strip() or None,
        question=question,
    )
    return {
        "file_id": str(record.get("id", "")).strip(),
        "analysis": str(analysis or "").strip()[:6000],
        "question": str(question or "").strip() or None,
    }
