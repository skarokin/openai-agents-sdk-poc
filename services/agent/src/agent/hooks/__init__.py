"""Global agent hooks and policies."""

from .deadline import (
    GLOBAL_DEADLINE_HOOK,
    SOFT_DEADLINE_MESSAGE,
    SOFT_DEADLINE_SECONDS,
    make_tool_enabled,
    make_tool_policy_guardrail,
    soft_deadline_model_filter,
)

__all__ = [
    "GLOBAL_DEADLINE_HOOK",
    "SOFT_DEADLINE_MESSAGE",
    "SOFT_DEADLINE_SECONDS",
    "make_tool_enabled",
    "make_tool_policy_guardrail",
    "soft_deadline_model_filter",
]
