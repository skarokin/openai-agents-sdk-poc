"""Out-of-band binding of function tools and subagent specs to the SDK."""

from collections.abc import Callable
from typing import Any

from agents import Agent, AgentToolStreamEvent, FunctionTool, function_tool

from agent.core.event_mapping import map_stream_event
from agent.core.models import AgentContext, EventSource, SubagentSpec
from agent.hooks import GLOBAL_DEADLINE_HOOK, soft_deadline_tool_guardrail


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
    implementation: Callable,
    *,
    name: str,
    needs_approval: bool,
) -> FunctionTool:
    return function_tool(
        implementation,
        name_override=name,
        needs_approval=needs_approval,
        tool_input_guardrails=[soft_deadline_tool_guardrail],
    )


def bind_subagent_tool(
    spec: SubagentSpec,
    *,
    name: str,
    tools: list[FunctionTool],
    model: Any,
    max_turns: int,
    context: AgentContext,
    needs_approval: bool,
) -> FunctionTool:
    agent = Agent[AgentContext](
        name=spec.agent_name,
        instructions=spec.instructions,
        model=model,
        tools=tools,
    )
    tool = agent.as_tool(
        tool_name=name,
        tool_description=spec.description,
        needs_approval=needs_approval,
        hooks=GLOBAL_DEADLINE_HOOK,
        max_turns=max_turns,
        on_stream=nested_stream_handler(context, name),
    )
    tool.tool_input_guardrails = [soft_deadline_tool_guardrail]
    return tool
