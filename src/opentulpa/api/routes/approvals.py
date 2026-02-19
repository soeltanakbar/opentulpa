"""Approval-related internal API routes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def register_approval_routes(
    app: FastAPI,
    *,
    get_approvals: Callable[[], Any],
    get_wake_queue: Callable[[], Any],
    get_agent_runtime: Callable[[], Any],
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Register approval endpoints and return shared decision helper."""

    async def _decide_approval_and_maybe_wake(
        *,
        approval_id: str,
        decision: str,
        actor_interface: str,
        actor_id: str,
    ) -> dict[str, Any]:
        broker = get_approvals()
        resolved = await broker.decide(
            approval_id=approval_id,
            decision=decision,
            actor_interface=actor_interface,
            actor_id=actor_id,
        )
        if bool(resolved.get("ok")) and str(resolved.get("status", "")).strip() == "approved":
            payload = {
                "type": "approval_event",
                "event_type": "approved",
                "customer_id": str(resolved.get("customer_id", "")).strip(),
                "thread_id": str(resolved.get("thread_id", "")).strip(),
                "approval_id": str(resolved.get("id", approval_id)).strip(),
                "payload": {
                    "approval_id": str(resolved.get("id", approval_id)).strip(),
                    "action_name": str(resolved.get("action_name", "")).strip(),
                    "summary": str(resolved.get("summary", "")).strip(),
                },
            }
            try:
                queue_id = await get_wake_queue().enqueue(payload)
                resolved["wake_queued"] = True
                resolved["wake_queue_id"] = queue_id
            except Exception:
                resolved["wake_queued"] = False
        return resolved

    @app.post("/internal/approvals/evaluate")
    async def internal_approvals_evaluate(request: Request) -> Any:
        broker = get_approvals()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        thread_id = str(body.get("thread_id", "")).strip()
        action_name = str(body.get("action_name", "")).strip()
        action_args = body.get("action_args") if isinstance(body.get("action_args"), dict) else {}
        origin_interface = str(body.get("origin_interface", "")).strip() or None
        origin_user_id = str(body.get("origin_user_id", "")).strip() or None
        origin_conversation_id = str(body.get("origin_conversation_id", "")).strip() or None
        if not customer_id or not thread_id or not action_name:
            return JSONResponse(
                status_code=400,
                content={"detail": "customer_id, thread_id, and action_name are required"},
            )
        decision = await broker.evaluate_action(
            customer_id=customer_id,
            thread_id=thread_id,
            action_name=action_name,
            action_args=action_args,
            origin_interface=origin_interface,
            origin_user_id=origin_user_id,
            origin_conversation_id=origin_conversation_id,
        )
        return decision

    @app.post("/internal/approvals/decide")
    async def internal_approvals_decide(request: Request) -> Any:
        body = await request.json()
        approval_id = str(body.get("approval_id", "")).strip()
        decision = str(body.get("decision", "")).strip().lower()
        actor_interface = str(body.get("actor_interface", "")).strip()
        actor_id = str(body.get("actor_id", "")).strip()
        if not approval_id or decision not in {"approve", "deny"} or not actor_interface or not actor_id:
            return JSONResponse(
                status_code=400,
                content={
                    "detail": (
                        "approval_id, decision (approve|deny), actor_interface, and actor_id are required"
                    )
                },
            )
        return await _decide_approval_and_maybe_wake(
            approval_id=approval_id,
            decision=decision,
            actor_interface=actor_interface,
            actor_id=actor_id,
        )

    @app.post("/internal/approvals/execute")
    async def internal_approvals_execute(request: Request) -> Any:
        broker = get_approvals()
        body = await request.json()
        approval_id = str(body.get("approval_id", "")).strip()
        customer_id = str(body.get("customer_id", "")).strip()
        if not approval_id or not customer_id:
            return JSONResponse(
                status_code=400, content={"detail": "approval_id and customer_id are required"}
            )
        agent_runtime = get_agent_runtime()
        if agent_runtime is None:
            return JSONResponse(status_code=503, content={"detail": "agent runtime unavailable"})

        inject_customer_id = {
            "memory_search",
            "memory_add",
            "uploaded_file_search",
            "uploaded_file_get",
            "uploaded_file_send",
            "web_image_send",
            "uploaded_file_analyze",
            "skill_list",
            "skill_get",
            "skill_upsert",
            "skill_delete",
            "directive_get",
            "directive_set",
            "directive_clear",
            "time_profile_get",
            "time_profile_set",
            "routine_list",
            "routine_create",
            "routine_delete",
            "automation_delete",
            "browser_use_run",
        }

        async def _approved_executor(action_name: str, action_args: dict[str, Any], cid: str) -> Any:
            await agent_runtime.start()
            tool_fn = agent_runtime._tools.get(action_name)
            if tool_fn is None:
                raise RuntimeError(f"unknown tool for approved execution: {action_name}")
            args = action_args if isinstance(action_args, dict) else {}
            if action_name in inject_customer_id:
                args = {**args, "customer_id": cid}
            return await tool_fn.ainvoke(args)

        try:
            result = await broker.execute_approved_action(
                approval_id=approval_id,
                customer_id=customer_id,
                executor=_approved_executor,
            )
        except Exception as exc:
            return JSONResponse(status_code=500, content={"detail": f"execute failed: {exc}"})
        if not bool(result.get("ok")):
            return JSONResponse(status_code=400, content=result)
        return result

    @app.get("/internal/approvals/{approval_id}")
    async def internal_approvals_get(approval_id: str) -> Any:
        broker = get_approvals()
        payload = broker.get(approval_id)
        if payload is None:
            return JSONResponse(status_code=404, content={"detail": "approval not found"})
        return {"ok": True, "approval": payload}

    return _decide_approval_and_maybe_wake
