from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from opentulpa.api.app import create_app


class _DummyTool:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(args)
        return {"ok": True, "echo": args}


class _DummyRuntime:
    def __init__(self) -> None:
        self.started = 0
        self.tool = _DummyTool()
        self._tools = {"dummy_action": self.tool}

    async def start(self) -> None:
        self.started += 1

    async def shutdown(self) -> None:
        return None

    def healthy(self) -> bool:
        return True


def test_approval_endpoints_lifecycle(tmp_path: Path) -> None:
    runtime = _DummyRuntime()
    app = create_app(agent_runtime=runtime)
    with TestClient(app) as client:
        evaluate = client.post(
            "/internal/approvals/evaluate",
            json={
                "customer_id": "cust_9",
                "thread_id": "chat-9",
                "action_name": "dummy_action",
                "action_args": {"x": 1},
                "origin_interface": "unknown",
                "origin_user_id": "99",
                "origin_conversation_id": "",
            },
        )
        assert evaluate.status_code == 200
        payload = evaluate.json()
        assert payload["gate"] == "require_approval"
        approval_id = payload["approval_id"]

        fetched = client.get(f"/internal/approvals/{approval_id}")
        assert fetched.status_code == 200
        assert fetched.json()["approval"]["status"] == "pending"

        wrong_actor = client.post(
            "/internal/approvals/decide",
            json={
                "approval_id": approval_id,
                "decision": "approve",
                "actor_interface": "unknown",
                "actor_id": "100",
            },
        )
        assert wrong_actor.status_code == 200
        assert wrong_actor.json()["ok"] is False
        assert wrong_actor.json()["reason"] == "unauthorized_actor"

        approved = client.post(
            "/internal/approvals/decide",
            json={
                "approval_id": approval_id,
                "decision": "approve",
                "actor_interface": "unknown",
                "actor_id": "99",
            },
        )
        assert approved.status_code == 200
        assert approved.json()["ok"] is True
        assert approved.json()["status"] == "approved"

        executed = client.post(
            "/internal/approvals/execute",
            json={"approval_id": approval_id, "customer_id": "cust_9"},
        )
        assert executed.status_code == 200
        assert executed.json()["ok"] is True
        assert runtime.tool.calls == [{"x": 1}]

        replay = client.post(
            "/internal/approvals/execute",
            json={"approval_id": approval_id, "customer_id": "cust_9"},
        )
        assert replay.status_code == 400
        assert replay.json()["error"] == "approval_already_executed"
