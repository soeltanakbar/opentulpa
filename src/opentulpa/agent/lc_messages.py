"""Compatibility exports for LangChain message classes across versions."""

from __future__ import annotations

try:
    # LangChain <1.0 style
    from langchain.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
except Exception:
    # LangChain 0.3+ / core package split
    from langchain_core.messages import (
        AIMessage,
        AnyMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

__all__ = [
    "AIMessage",
    "AnyMessage",
    "HumanMessage",
    "SystemMessage",
    "ToolMessage",
]
