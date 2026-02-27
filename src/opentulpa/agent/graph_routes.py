"""Graph routing functions extracted from graph_builder."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END

from opentulpa.agent.lc_messages import AIMessage
from opentulpa.agent.models import AgentState


def route_after_agent(state: AgentState) -> Literal["validate_tools", "claim_check"]:
    messages = state.get("messages", [])
    if not messages:
        return "claim_check"
    last = messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "validate_tools"
    return "claim_check"


def route_after_validate(state: AgentState) -> Literal["tools", "agent"]:
    if state.get("tool_validation_passed", True):
        return "tools"
    return "agent"


def route_after_tools(state: AgentState) -> Literal["agent", END]:
    if bool(state.get("approval_handoff", False)):
        return END
    return "agent"


def route_after_claim_check(state: AgentState) -> Literal["agent", END]:
    if bool(state.get("claim_check_needs_retry", False)):
        return "agent"
    return END
