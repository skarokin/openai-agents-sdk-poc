"""Global agent hooks."""

from .deadline import (
    GLOBAL_DEADLINE_HOOK,
    SOFT_DEADLINE_MESSAGE,
    SOFT_DEADLINE_SECONDS,
    soft_deadline_model_filter,
    soft_deadline_tool_guardrail,
)

__all__ = [
    "GLOBAL_DEADLINE_HOOK",
    "SOFT_DEADLINE_MESSAGE",
    "SOFT_DEADLINE_SECONDS",
    "soft_deadline_model_filter",
    "soft_deadline_tool_guardrail",
]
