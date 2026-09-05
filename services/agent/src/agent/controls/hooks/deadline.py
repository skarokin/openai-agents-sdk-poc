"""Soft-deadline enforcement via Strands hooks."""

import logging
import time
from typing import Any

from strands.hooks import BeforeModelCallEvent, HookProvider, HookRegistry


logger = logging.getLogger(__name__)


SOFT_DEADLINE_SECONDS = 300.0
SOFT_DEADLINE_MESSAGE = (
    "The soft deadline has expired. Do not call this or any other tool. "
    "Return the best answer you can using information already available."
)


class SoftDeadlineHook(HookProvider):
    """If the deadline has passed, steer the next model call to finish without tools."""

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeModelCallEvent, self.on_before_model)

    def on_before_model(self, event: BeforeModelCallEvent) -> None:
        deadline = event.agent.state.get("deadline_epoch_seconds")
        if deadline is None:
            return

        if time.time() < float(deadline):
            return

        request_id = event.agent.state.get("request_id") or "-"
        identity = event.agent.state.get("identity") or {}
        logger.warning(
            "soft deadline exceeded at before_model:%s",
            event.agent.name,
            extra={
                "request_id": request_id,
                "subject_id": identity.get("subject_id", "-"),
            },
        )
        event.agent.messages.append(
            {
                "role": "user",
                "content": [{"text": SOFT_DEADLINE_MESSAGE}],
            }
        )


GLOBAL_DEADLINE_HOOK = SoftDeadlineHook()
