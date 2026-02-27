"""Wake-event classifier helper for runtime."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from opentulpa.agent.lc_messages import HumanMessage, SystemMessage
from opentulpa.agent.result_models import WakeEventDecision


async def classify_wake_event(
    *,
    classifier_model: Any,
    extract_json_object: Callable[[str], dict[str, Any] | None],
    customer_id: str,
    event_label: str,
    payload: dict[str, Any],
) -> WakeEventDecision:
    """Let the model decide whether a wake event should interrupt the user now."""
    try:
        response = await classifier_model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You classify background assistant events.\n"
                        "Return strict JSON with keys: notify_user (bool), reason (string).\n"
                        "Use notify_user=true only when immediate user attention is required."
                    )
                ),
                HumanMessage(
                    content=(
                        f"customer_id={customer_id}\n"
                        f"event_label={event_label}\n"
                        f"payload={json.dumps(payload, ensure_ascii=False)[:5000]}"
                    )
                ),
            ]
        )
        raw = response.content if hasattr(response, "content") else str(response)
        raw_text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
        parsed = extract_json_object(raw_text) or {}
        return WakeEventDecision(
            notify_user=bool(parsed.get("notify_user", False)),
            reason=str(parsed.get("reason", "")).strip()[:500],
        )
    except Exception as exc:
        return WakeEventDecision(notify_user=False, reason=f"classifier_error:{exc}")
