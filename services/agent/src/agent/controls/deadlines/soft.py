"""Soft-deadline enforcement, implemented as a hook on BeforeModelCallEvent"""

import logging
import time
from typing import Any

from strands.hooks import BeforeModelCallEvent, HookProvider, HookRegistry

from .constants import SOFT_DEADLINE_MESSAGE


logger = logging.getLogger(__name__)


def _request_meta(event: BeforeModelCallEvent) -> tuple[str, str]:
    request_id = (
        event.invocation_state.get("request_id")
        or event.agent.state.get("request_id")
        or "-"
    )
    identity = (
        event.invocation_state.get("identity")
        or event.agent.state.get("identity")
        or {}
    )
    if isinstance(identity, dict):
        subject_id = identity.get("subject_id", "-")
    else:
        subject_id = getattr(identity, "subject_id", "-")
    return str(request_id), str(subject_id)


class SoftDeadlineHook(HookProvider):
    """
    Shared soft-deadline observer for root and nested agents.

    When soft deadline has passed (and hard has not), cancel the model call
    with a clean assistant message instead of streaming a partial response.
    Hard cancel is owned by the shared cancel_signal / watchdog.
    """

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeModelCallEvent, self.on_before_model)

    @staticmethod
    def _deadline_from(event: BeforeModelCallEvent, key: str) -> float | None:
        value = event.invocation_state.get(key)
        if value is None:
            value = event.agent.state.get(key)
        if value is None:
            return None
        return float(value)

    def on_before_model(self, event: BeforeModelCallEvent) -> None:
        soft = self._deadline_from(event, "soft_deadline_epoch_seconds")
        if soft is None:
            return

        now = time.time()
        if now < soft:
            return

        hard = self._deadline_from(event, "hard_deadline_epoch_seconds")
        if hard is not None and now >= hard:
            # if hard deadline is reached, dont soft-cancel; let hard-cancel take effect
            # this probably wont happen but just in case
            return

        request_id, subject_id = _request_meta(event)
        logger.warning(
            "soft deadline exceeded at before_model:%s",
            event.agent.name,
            extra={"request_id": request_id, "subject_id": subject_id},
        )

        # event.cancel will stop all tool calls, and present the agent with the soft deadline message
        event.cancel = SOFT_DEADLINE_MESSAGE


GLOBAL_DEADLINE_HOOK = SoftDeadlineHook()
