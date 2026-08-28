"""Soft-deadline enforcement only. Nested runs get this hook via as_tool(hooks=...)."""

import logging

from agents import (
    RunContextWrapper,
    RunHooks,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrailData,
    tool_input_guardrail,
)
from agents.run_config import CallModelData, ModelInputData

from agent.core.models import AgentContext, EventSource, GuardrailTripped

logger = logging.getLogger(__name__)

SOFT_DEADLINE_SECONDS = 120.0
SOFT_DEADLINE_MESSAGE = (
    "The soft deadline has expired. Do not call this or any other tool. "
    "Return the best answer you can using information already available."
)


@tool_input_guardrail(name="soft_deadline")
async def soft_deadline_tool_guardrail(
    data: ToolInputGuardrailData,
) -> ToolGuardrailFunctionOutput:
    """Block tool execution after the deadline; emit a stream event if a sink exists."""

    context: AgentContext = data.context.context
    if not context.deadline_exceeded:
        return ToolGuardrailFunctionOutput.allow()

    await context.event_sink.emit(
        GuardrailTripped(
            guardrail_name="soft_deadline",
            kind="tool_input",
            message=SOFT_DEADLINE_MESSAGE,
        ),
        source=EventSource(
            agent_name=data.agent.name,
            invocation_id=data.context.tool_call_id,
        ),
    )
    return ToolGuardrailFunctionOutput.reject_content(
        SOFT_DEADLINE_MESSAGE,
        output_info={"reason": "soft_deadline"},
    )


def soft_deadline_model_filter(
    data: CallModelData[AgentContext],
) -> ModelInputData:
    """Tell each model call to finish without tools after the deadline."""

    context = data.context
    if context is None or not context.deadline_exceeded:
        return data.model_data

    instructions = data.model_data.instructions or ""
    return ModelInputData(
        input=data.model_data.input,
        instructions=f"{instructions}\n\n{SOFT_DEADLINE_MESSAGE}".strip(),
    )


class SoftDeadlineHook(RunHooks[AgentContext]):
    """Observe the deadline at SDK lifecycle checkpoints, including nested as_tool runs."""

    @staticmethod
    def _check(context: RunContextWrapper[AgentContext], checkpoint: str) -> None:
        if context.context.deadline_exceeded:
            logger.warning(
                "soft deadline exceeded at %s",
                checkpoint,
                extra={
                    "request_id": context.context.request_id,
                    "subject_id": context.context.identity.subject_id,
                },
            )

    async def on_agent_start(self, context, agent) -> None:
        self._check(context, f"agent_start:{agent.name}")

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        del system_prompt, input_items
        self._check(context, f"llm_start:{agent.name}")

    async def on_tool_start(self, context, agent, tool) -> None:
        self._check(context, f"tool_start:{agent.name}:{tool.name}")

    async def on_tool_end(self, context, agent, tool, result) -> None:
        del result
        self._check(context, f"tool_end:{agent.name}:{tool.name}")


GLOBAL_DEADLINE_HOOK = SoftDeadlineHook()
