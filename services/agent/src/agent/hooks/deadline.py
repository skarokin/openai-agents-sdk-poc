"""Soft-deadline enforcement via RunHooks. Nested runs get this hook via as_tool(hooks=...)."""

import logging

from agents import RunHooks

from agent.core.models import AgentContext

logger = logging.getLogger(__name__)

SOFT_DEADLINE_SECONDS = 300.0
SOFT_DEADLINE_MESSAGE = (
    "The soft deadline has expired. Do not call this or any other tool. "
    "Return the best answer you can using information already available."
)


class SoftDeadlineHook(RunHooks[AgentContext]):
    """If the deadline has passed, steer the next model call to finish without tools."""

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        if not context.context.deadline_exceeded:
            return

        logger.warning(
            "soft deadline exceeded at llm_start:%s",
            agent.name,
            extra={
                "request_id": context.context.request_id,
                "subject_id": context.context.identity.subject_id,
            },
        )

        if input_items is None:
            return

        input_items.append(
            {
                "role": "developer",
                "content": SOFT_DEADLINE_MESSAGE,
            }
        )


GLOBAL_DEADLINE_HOOK = SoftDeadlineHook()
