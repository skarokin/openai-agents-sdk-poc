"""Global soft-deadline policy for root and nested agent runs."""

import logging
from collections.abc import Callable
from typing import Any, cast

from agents import (
    AgentBase,
    RunContextWrapper,
    RunHooks,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrail,
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


def _roles_allowed(context: AgentContext, allowed_roles: frozenset[str]) -> bool:
    return not allowed_roles or bool(context.identity.roles & allowed_roles)


def make_tool_enabled(
    allowed_roles: frozenset[str],
) -> Callable[[RunContextWrapper[Any], AgentBase[Any]], bool]:
    """Hide tools the caller cannot use."""

    def is_enabled(
        context: RunContextWrapper[Any],
        agent: AgentBase[Any],
    ) -> bool:
        del agent
        agent_context = cast(AgentContext, context.context)
        return _roles_allowed(agent_context, allowed_roles)

    return is_enabled


def make_tool_policy_guardrail(
    tool_name: str,
    allowed_roles: frozenset[str],
) -> ToolInputGuardrail[AgentContext]:
    """Recheck deadline and RBAC immediately before tool execution."""

    @tool_input_guardrail(name=f"{tool_name}_policy")
    async def policy(
        data: ToolInputGuardrailData,
    ) -> ToolGuardrailFunctionOutput:
        context: AgentContext = data.context.context
        if context.deadline_exceeded:
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
                output_info={"tool": tool_name, "reason": "soft_deadline"},
            )

        if not _roles_allowed(context, allowed_roles):
            message = f"The caller is not authorized to use the {tool_name} tool."
            return ToolGuardrailFunctionOutput.reject_content(
                message,
                output_info={"tool": tool_name, "reason": "forbidden"},
            )

        return ToolGuardrailFunctionOutput.allow()

    return policy


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
    """Observe the deadline at SDK lifecycle checkpoints."""

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
