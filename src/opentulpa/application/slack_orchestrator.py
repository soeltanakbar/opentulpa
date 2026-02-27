"""Application-layer orchestration for Slack APIs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opentulpa.application.contracts import ApplicationResult


class SlackOrchestratorResult(ApplicationResult[dict[str, object]]):
    """Normalized route-friendly result payload."""


class SlackOrchestrator:
    """Owns Slack endpoint business rules independent of FastAPI transport."""

    def __init__(
        self,
        *,
        get_slack: Callable[[], Any],
        has_write_consent: Callable[[str], bool],
        grant_write_consent: Callable[[str], None],
    ) -> None:
        self._get_slack = get_slack
        self._has_write_consent = has_write_consent
        self._grant_write_consent = grant_write_consent

    async def list_channels(self, *, limit: int, cursor: str) -> SlackOrchestratorResult:
        result = await self._get_slack().list_channels(limit=limit, cursor=cursor or None)
        return SlackOrchestratorResult(status_code=200, payload=result)

    async def channel_history(
        self,
        *,
        channel_id: str,
        limit: int,
        cursor: str,
    ) -> SlackOrchestratorResult:
        result = await self._get_slack().channel_history(channel_id, limit=limit, cursor=cursor or None)
        return SlackOrchestratorResult(status_code=200, payload=result)

    def grant_consent(self, *, customer_id: str | None, scope: str) -> SlackOrchestratorResult:
        safe_customer_id = str(customer_id or "").strip()
        safe_scope = str(scope or "").strip()
        if not safe_customer_id:
            return SlackOrchestratorResult(status_code=400, payload={"detail": "customer_id required"})
        if safe_scope != "write":
            return SlackOrchestratorResult(status_code=400, payload={"detail": "scope must be 'write'"})
        self._grant_write_consent(safe_customer_id)
        return SlackOrchestratorResult(
            status_code=200,
            payload={"ok": True, "message": "Slack write consent granted."},
        )

    async def post_message(
        self,
        *,
        customer_id: str | None,
        channel_id: str,
        text: str,
        thread_ts: str | None,
    ) -> SlackOrchestratorResult:
        safe_customer_id = str(customer_id or "").strip()
        if not safe_customer_id:
            return SlackOrchestratorResult(
                status_code=400,
                payload={
                    "ok": False,
                    "error": "customer_id required",
                    "detail": "customer_id required",
                },
            )
        if not channel_id or not text:
            return SlackOrchestratorResult(
                status_code=400,
                payload={"detail": "channel_id and text required"},
            )
        if not self._has_write_consent(safe_customer_id):
            return SlackOrchestratorResult(
                status_code=403,
                payload={
                    "ok": False,
                    "error": "consent_required",
                    "message": (
                        "The user has not granted permission to post to Slack. "
                        "Ask the user to confirm they allow the agent to post to Slack on their behalf; "
                        "once they agree, use slack_grant_write_consent and then try posting again."
                    ),
                },
            )
        result = await self._get_slack().post_message(channel_id, text, thread_ts=thread_ts)
        return SlackOrchestratorResult(status_code=200, payload=result)
