"""Out-of-band binding of function tools and subagent specs to the SDK."""

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from agents import Agent, AgentToolStreamEvent, FunctionTool, function_tool

from agent.core.event_mapping import map_stream_event
from agent.core.models import AgentContext, EventSource, SubagentSpec
from agent.hooks import GLOBAL_DEADLINE_HOOK


def nested_stream_handler(context: AgentContext, fallback_invocation_id: str):
    """Forward nested agent stream events onto the parent EventSink."""

    previous_agents: dict[str, str] = {}

    async def on_stream(payload: AgentToolStreamEvent) -> None:
        event = payload["event"]
        agent = payload["agent"]
        tool_call = payload["tool_call"]
        invocation_id = getattr(tool_call, "call_id", None) or fallback_invocation_id
        previous = previous_agents.get(invocation_id)

        for normalized in map_stream_event(event, previous_agent_name=previous):
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


def bind_function_tool(
    implementation: FunctionTool | Callable,
    *,
    name: str,
    is_enabled: bool,
) -> FunctionTool:
    """Bind a function tool to the agent."""

    if isinstance(implementation, FunctionTool):
        return replace(implementation, name=name, is_enabled=is_enabled)
    return function_tool(
        implementation,
        name_override=name,
        is_enabled=is_enabled,
    )


def bind_subagent_tool(
    spec: SubagentSpec,
    *,
    name: str,
    tools: list[FunctionTool],
    model: Any,
    max_turns: int,
    context: AgentContext,
    is_enabled: bool,
) -> FunctionTool:
    """Bind an agent-as-tool to the parent agent."""

    agent = Agent[AgentContext](
        name=spec.agent_name,
        instructions=spec.instructions,
        model=model,
        tools=tools,
    )

    return agent.as_tool(
        tool_name=name,
        tool_description=spec.description,
        needs_approval=spec.needs_approval,
        is_enabled=is_enabled,
        hooks=GLOBAL_DEADLINE_HOOK,
        max_turns=max_turns,
        on_stream=nested_stream_handler(context, name),
    )
