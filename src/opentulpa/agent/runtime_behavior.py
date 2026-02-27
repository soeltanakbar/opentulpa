"""Behavior-log helper for runtime instrumentation."""

from __future__ import annotations

import json
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log_behavior_event(
    *,
    behavior_log_enabled: bool,
    event: str,
    fields: dict[str, Any],
    behavior_log_path: Path | Any,
    behavior_log_lock: Any | None,
) -> None:
    if not behavior_log_enabled:
        return
    event_name = str(event or "").strip()
    if not event_name:
        return
    payload: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event_name,
    }
    for key, value in fields.items():
        safe_key = str(key or "").strip()
        if not safe_key:
            continue
        payload[safe_key] = value
    serialized = json.dumps(payload, ensure_ascii=False, default=str)
    if not isinstance(behavior_log_path, Path):
        return
    with suppress(Exception):
        behavior_log_path.parent.mkdir(parents=True, exist_ok=True)
    if behavior_log_lock is None:
        with suppress(Exception), behavior_log_path.open("a", encoding="utf-8") as f:
            f.write(serialized + "\n")
        return
    with suppress(Exception), behavior_log_lock, behavior_log_path.open("a", encoding="utf-8") as f:
        f.write(serialized + "\n")
