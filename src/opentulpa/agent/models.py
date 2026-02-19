"""Agent runtime state models."""

from __future__ import annotations

from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from opentulpa.agent.lc_messages import AnyMessage


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    customer_id: str
    thread_id: str
    active_skill_query: str
    active_skill_context: str
    active_skill_names: list[str]
    tool_validation_passed: bool
    guardrail_has_executable_calls: bool
    guardrail_allowed_call_ids: list[str]
    guardrail_feedback_messages: list[AnyMessage]
    tool_error_count: int
    last_tool_error: str
