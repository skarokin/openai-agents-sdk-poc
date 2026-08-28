"""Create per-request agent graphs from global configuration and identity."""

import logging
from typing import Any

from agents import Agent, AgentToolStreamEvent, FunctionTool, function_tool

from agent.hooks import (
    GLOBAL_DEADLINE_HOOK,
    make_tool_enabled,
    make_tool_policy_guardrail,
)
from agent.tools import approval_demo, calculator

from .config import ServiceConfig, ToolConfig
from .event_mapping import map_stream_event
from .models import AgentContext, EventSource

logger = logging.getLogger(__name__)


class AgentFactory:
    """Build an agent graph containing only tools allowed for the caller."""

    def __init__(self, config: ServiceConfig, model: Any | None = None):
        self._config = config
        self._model = model or config.agent.model

    @property
    def max_turns(self) -> int:
        return self._config.agent.max_turns

    @staticmethod
    def _authorized(settings: ToolConfig, context: AgentContext) -> bool:
        return settings.enabled and (
            not settings.allowed_roles
            or bool(settings.allowed_roles & context.identity.roles)
        )

    def _function_tool(
        self,
        name: str,
        implementation,
        context: AgentContext,
    ) -> FunctionTool | None:
        settings = self._config.tools.tools[name]
        if not self._authorized(settings, context):
            return None
        return function_tool(
            implementation,
            name_override=name,
            is_enabled=make_tool_enabled(settings.allowed_roles),
            needs_approval=settings.requires_approval,
            tool_input_guardrails=[
                make_tool_policy_guardrail(name, settings.allowed_roles)
            ],
        )

    def _nested_stream_handler(
        self,
        context: AgentContext,
        fallback_invocation_id: str,
    ):
        previous_agents: dict[str, str] = {}

        async def on_stream(payload: AgentToolStreamEvent) -> None:
            event = payload["event"]
            agent = payload["agent"]
            tool_call = payload["tool_call"]
            invocation_id = (
                getattr(tool_call, "call_id", None) or fallback_invocation_id
            )
            previous = previous_agents.get(invocation_id)
            for normalized in map_stream_event(
                event,
                previous_agent_name=previous,
            ):
                await context.event_sink.emit(
                    normalized,
                    source=EventSource(
                        agent_name=agent.name,
                        invocation_id=invocation_id,
                        parent_invocation_id=context.request_id,
                        kind="agent_tool",
                    ),
                )
            previous_agents[invocation_id] = agent.name

        return on_stream

    def _subagent_tool(
        self,
        name: str,
        agent: Agent[AgentContext],
        context: AgentContext,
    ) -> FunctionTool | None:
        settings = self._config.tools.tools[name]
        if not self._authorized(settings, context):
            return None

        tool = agent.as_tool(
            tool_name=name,
            tool_description=(
                "Delegate a focused task to a subagent that can calculate and "
                "demonstrate nested approval workflows."
            ),
            is_enabled=make_tool_enabled(settings.allowed_roles),
            needs_approval=settings.requires_approval,
            hooks=GLOBAL_DEADLINE_HOOK,
            max_turns=self._config.agent.max_turns,
            on_stream=self._nested_stream_handler(context, name),
        )
        tool.tool_input_guardrails = [
            make_tool_policy_guardrail(name, settings.allowed_roles)
        ]
        return tool

    def create(self, context: AgentContext) -> Agent[AgentContext]:
        calculator_tool = self._function_tool("calculator", calculator, context)
        approval_tool = self._function_tool(
            "approval_demo",
            approval_demo,
            context,
        )
        shared_tools = [
            tool for tool in (calculator_tool, approval_tool) if tool is not None
        ]

        subagent_instructions = (
            "You are a focused subagent. Use the calculator for arithmetic. "
            "When explicitly asked to demonstrate approval, call approval_demo. "
            "Return a concise answer to the parent agent."
        )
        protected_subagent = Agent[AgentContext](
            name="Protected Subagent",
            instructions=subagent_instructions,
            model=self._model,
            tools=list(shared_tools),
        )
        open_subagent = Agent[AgentContext](
            name="Open Subagent",
            instructions=subagent_instructions,
            model=self._model,
            tools=list(shared_tools),
        )

        nested_tools = [
            tool
            for tool in (
                self._subagent_tool(
                    "protected_subagent",
                    protected_subagent,
                    context,
                ),
                self._subagent_tool("open_subagent", open_subagent, context),
            )
            if tool is not None
        ]
        tools: list[Any] = [*shared_tools, *nested_tools]
        logger.info(
            "agent graph created",
            extra={
                "subject_id": context.identity.subject_id,
                "tool_names": [tool.name for tool in tools],
            },
        )
        return Agent[AgentContext](
            name=self._config.agent.name,
            instructions=self._config.agent.instructions,
            model=self._model,
            tools=tools,
        )
