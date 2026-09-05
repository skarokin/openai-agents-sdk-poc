"""Deadline policy: soft model-call cancel + hard cancel_signal watchdog."""

from .constants import (
    HARD_DEADLINE_SECONDS,
    SOFT_DEADLINE_MESSAGE,
    SOFT_DEADLINE_SECONDS,
)
from .hard import start_hard_deadline_watchdog
from .soft import GLOBAL_DEADLINE_HOOK, SoftDeadlineHook

__all__ = [
    "GLOBAL_DEADLINE_HOOK",
    "HARD_DEADLINE_SECONDS",
    "SOFT_DEADLINE_MESSAGE",
    "SOFT_DEADLINE_SECONDS",
    "SoftDeadlineHook",
    "start_hard_deadline_watchdog",
]
