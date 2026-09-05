"""
Shared interrupt reason shapes for nested HITL bubbling.

HITL is a first-class feature of Strands so this is just a helper for our desired behavior of
nested HITL bubbling.
"""

from collections.abc import Mapping
from typing import Any


# Bubbled from a nested agent-as-tool when nested agent pauses for HITL
NESTED_HITL_REASON_KEY = "nested_hitl"


def nested_hitl_reason(
    *,
    agent_name: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return {
        NESTED_HITL_REASON_KEY: True,
        "agent_name": agent_name,
        "tool_name": tool_name,
        "arguments": arguments,
    }


def is_nested_hitl_reason(reason: Any) -> bool:
    return isinstance(reason, Mapping) and bool(reason.get(NESTED_HITL_REASON_KEY))
