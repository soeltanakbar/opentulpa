"""Agent runtime state models."""

from __future__ import annotations

from typing import Annotated

from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    customer_id: str
    thread_id: str
    active_skill_query: str
    active_skill_context: str
    active_skill_names: list[str]
    tool_validation_passed: bool
    tool_error_count: int
    last_tool_error: str
