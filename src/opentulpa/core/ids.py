"""Compact, LLM-friendly ID generation utilities."""

from __future__ import annotations

import re
import uuid

_PREFIX_RE = re.compile(r"^[a-z][a-z0-9_-]{0,20}$")
_BASE36_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


def _to_base36(value: int) -> str:
    if value <= 0:
        return "0"
    out: list[str] = []
    n = int(value)
    while n:
        n, rem = divmod(n, 36)
        out.append(_BASE36_ALPHABET[rem])
    return "".join(reversed(out))


def _uuid8_like_hex() -> str:
    """
    Build a UUID with version nibble set to 8 (UUIDv8-like layout).

    Python <3.14 does not expose uuid.uuid8(), so we derive one from UUID4 bytes.
    """
    raw = bytearray(uuid.uuid4().bytes)
    raw[6] = (raw[6] & 0x0F) | 0x80  # version 8
    raw[8] = (raw[8] & 0x3F) | 0x80  # RFC 4122 variant
    return uuid.UUID(bytes=bytes(raw)).hex


def _base36_entropy(chars: int) -> str:
    token = _to_base36(int(_uuid8_like_hex(), 16))
    return token[-chars:].rjust(chars, "0")


def new_short_id(prefix: str, *, suffix_chars: int = 6) -> str:
    """
    Return compact IDs that are easy for an LLM to preserve exactly.

    Format: <prefix>_<random_tail>
    Example: task_1k3k4a
    """
    safe_prefix = str(prefix or "").strip().lower()
    if not _PREFIX_RE.match(safe_prefix):
        raise ValueError("prefix must match ^[a-z][a-z0-9_-]{0,20}$")
    size = max(4, min(12, int(suffix_chars)))
    return f"{safe_prefix}_{_base36_entropy(size)}"
