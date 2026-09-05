"""Deadline constants shared by soft hook and hard watchdog."""

SOFT_DEADLINE_SECONDS = 300.0
HARD_DEADLINE_SECONDS = 330.0

SOFT_DEADLINE_MESSAGE = (
    "The soft deadline has expired. Stopping with the best answer available "
    "from work completed so far."
)
