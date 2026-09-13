"""Create per-request agents from global tool config and caller identity."""

import asyncio
import importlib
import logging
import re
from typing import Any

from strands import Agent, tool
from strands.models import Model
from strands.models.openai import OpenAIModel
from strands.session.file_session_manager import FileSessionManager
from strands.tools.mcp import MCPClient
from strands.types.tools import ToolContext
from strands.vended_interventions.hitl import HumanInTheLoop

from agent.config import ServiceConfig, ToolConfig, hitl_allowed_tools
from agent.controls.deadlines import GLOBAL_DEADLINE_HOOK
from agent.controls.hitl import nested_hitl_reason
from agent.core.event_mapping import (
    SUBAGENT_EVENT_KEY,
    final_output_text,
    is_tool_activity_event,
    tool_info_from_interrupt,
)
from agent.core.models import AgentContext, SubagentSpec
from agent.core.runtime import create_trace_attributes
from agent.core.sessions import ROOT_AGENT_ID, make_file_session_manager

logger = logging.getLogger(__name__)


def _load_symbol(import_path: str) -> Any:
    module_name, separator, attr = import_path.partition(":")
    if not separator or not attr:
        raise ValueError(f"Tool import_path must be 'module:attr', got {import_path}")

    return getattr(importlib.import_module(module_name), attr)


def _has_role(context: AgentContext, allowed_roles: frozenset[str]) -> bool:
    return bool(context.identity.roles & allowed_roles)


def _mcp_tool_filters(settings: ToolConfig, context: AgentContext):
    """Role-gate MCP child tools via Strands ToolFilters."""

    allowed: list[str | re.Pattern[str]] = []
    for child_name, child in settings.tools.items():
        roles = child.allowed_roles or settings.allowed_roles
        if _has_role(context, roles):
            allowed.append(child_name)

    if not allowed:
        return {"allowed": []}

    return {"allowed": allowed}


def _nested_interrupt_reason(
    interrupt: Any,
    *,
    agent_name: str,
) -> Any:
    """
    Tag bubbled nested interrupts so UI can show the subagent, not the root.

    Native HITL keeps a prompt string; we wrap it with agent_name plus tool
    name/args parsed from the interrupt itself (no tool_use_message).
    """

    reason = interrupt.reason
    if isinstance(reason, dict) and reason.get("requires_auth"):
        return {**reason, "agent_name": agent_name}

    tool_name, arguments = tool_info_from_interrupt(interrupt)
    return nested_hitl_reason(
        agent_name=agent_name,
        tool_name=tool_name,
        arguments=arguments,
    )


def openai_model(model: str | Model) -> Model:
    """Build an OpenAI Strands model, or pass through an existing Model."""

    if isinstance(model, Model):
        return model

    return OpenAIModel(
        model_id=model,
        params={
            "reasoning_effort": "none"
        }
    )


def build_subagent(
    agent: Agent,
    *,
    agent_id: str,
    session_manager: FileSessionManager,
    hitl_tools: frozenset[str] | set[str] = frozenset(),
) -> Agent:
    """
    Rebinds a self-defined Agent with non-negotiable dependencies. This is used for agent-as-tools.

    Always applied:
    - shared parent session_manager (interrupt/session continuity)
    - GLOBAL_DEADLINE_HOOK (deadlines must be enforced on all agents)
    - HITL intervention for tools that appear in both the agent and hitl_tools
    - stable agent_id (tool name) for session isolation within the shared manager
    - ... this list will grow ...

    If anything is added to the main agent that must be applied to all subagents, edit this function
    """

    tools = [agent.tool_registry.registry[name] for name in agent.tool_names]
    nested_hitl = frozenset(hitl_tools) & frozenset(agent.tool_names)

    return Agent(
        agent_id=agent_id,
        name=agent.name,
        description=agent.description,
        system_prompt=agent.system_prompt,
        model=agent.model,
        tools=tools,
        session_manager=session_manager,
        interventions=(
            [HumanInTheLoop(allowed_tools=hitl_allowed_tools(nested_hitl))]
            if nested_hitl
            else None
        ),
        hooks=[GLOBAL_DEADLINE_HOOK],
        callback_handler=None,
    )


def as_subagent_tool(
    agent: Agent,
    *,
    name: str,
    session_manager: FileSessionManager,
    hitl_tools: frozenset[str] | set[str] = frozenset(),
    description: str | None = None,
) -> Any:
    """
    Single place to wrap Agents as tools to ensure consistent behavior across subagents.
    """

    # *ALL* subagents will be built via this function to ensure consistency of the Agent object 
    nested = build_subagent(
        agent,
        agent_id=name,
        session_manager=session_manager,
        hitl_tools=hitl_tools,
    )
    display_name = nested.name or name
    tool_description = description or nested.description or ""

    # the actual subagent execution logic - *ALL* subagents will be ran via this function
    # if the actual *running* of subagents needs to change, edit this function
    async def run(input: str, tool_context: ToolContext):
        # forward all invocation state from the parent
        invocation_state = {
            key: value
            for key, value in tool_context.invocation_state.items()
            if key != "agent"
        }
        cancel_signal = (
            invocation_state.get("cancel_signal") or tool_context.cancel_signal
        )

        # if subagent has an active interrupt, we need to bubble down the response that the user
        # provided to the main agent back down to the subagent
        if nested._interrupt_state.activated:
            responses = []
            for interrupt in nested._interrupt_state.interrupts.values():
                response = tool_context.interrupt(
                    interrupt.id,
                    reason=_nested_interrupt_reason(
                        interrupt,
                        agent_name=display_name,
                    ),
                )
                responses.append(
                    {
                        "interruptResponse": {
                            "interruptId": interrupt.id,
                            "response": response,
                        }
                    }
                )
            prompt: Any = responses
        else:
            prompt = input

        result = None
        async for event in nested.stream_async(
            prompt,
            invocation_state=invocation_state,
            cancel_signal=cancel_signal,
        ):
            if "result" in event:
                result = event["result"]
                continue

            # we only care about two events from the subagent:
            # - tool activity
            # - errors (this should be implemented later)
            if is_tool_activity_event(event):
                yield {
                    SUBAGENT_EVENT_KEY: True,
                    "agent_name": display_name,
                    "event": dict(event),
                }

        if result is None:
            raise RuntimeError("nested agent stream ended without a result")

        # if subagent raised an interrupt, we need to bubble this up to the parent agent
        # so that it can show the user the interrupt
        if result.stop_reason == "interrupt" and result.interrupts:
            for interrupt in result.interrupts:
                tool_context.interrupt(
                    interrupt.id,
                    reason=_nested_interrupt_reason(
                        interrupt,
                        agent_name=display_name,
                    ),
                )
            raise RuntimeError("nested interrupt should have raised")

        # last yield is the tool result string (Strands async-gen contract)
        yield final_output_text(result.message) or str(result)

    decorated = tool(name=name, description=tool_description, context=True)(run)
    # this just allows tests to grab the nested Agent object
    decorated._nested_agent = nested  # type: ignore[attr-defined]

    return decorated


class AgentFactory:
    """Returns a Strands Agent with role-gated tools, MCP, and interventions."""

    def __init__(
        self,
        config: ServiceConfig,
        model: str | Model | None = None,
    ):
        self._config = config
        self._mcp_clients: dict[str, MCPClient] = {}
        self._mcp_lock = asyncio.Lock()
        self._model = openai_model(model if model is not None else config.agent.model)

    @property
    def max_turns(self) -> int:
        return self._config.agent.max_turns

    @property
    def hitl_tools(self) -> frozenset[str]:
        return self._config.hitl_tools

    @property
    def mcp_hitl_tools(self) -> frozenset[str]:
        return self._config.mcp_hitl_tools

    async def _connect_mcp(self, context: AgentContext) -> list[MCPClient]:
        """Startup and connect to all MCP the caller is allowed to use."""
        wanted = [
            (name, settings)
            for name, settings in self._config.tools.items()
            if settings.type == "mcp"
            and settings.enabled
            and _has_role(context, settings.allowed_roles)
        ]
        clients: list[MCPClient] = []

        async with self._mcp_lock:
            for name, settings in wanted:
                filters = _mcp_tool_filters(settings, context)
                cache_key = f"{name}:{sorted(filters.get('allowed') or [])}"
                client = self._mcp_clients.get(cache_key)
                if client is None:
                    create = _load_symbol(settings.import_path)
                    client = create(tool_filters=filters)
                    self._mcp_clients[cache_key] = client
                clients.append(client)

        return clients

    async def close(self) -> None:
        async with self._mcp_lock:
            for client in self._mcp_clients.values():
                client.stop(None, None, None)
            self._mcp_clients.clear()

    def _hitl_intervention(self) -> HumanInTheLoop | None:
        required = self._config.all_hitl_tools
        if not required:
            return None
        return HumanInTheLoop(allowed_tools=hitl_allowed_tools(required))

    async def create(self, context: AgentContext) -> Agent:
        session_manager = make_file_session_manager(
            context.session_id,
            context.identity.subject_id,
        )

        function_tools: dict[str, Any] = {}
        subagent_entries: list[tuple[str, SubagentSpec]] = []

        # build tool catalog from config, only including tools that are enabled and allowed for the caller's role
        # skips MCP and subagents which need to be handled differently
        for name, settings in self._config.tools.items():
            if not settings.enabled or settings.type == "mcp" or not _has_role(context, settings.allowed_roles):
                continue

            implementation = _load_symbol(settings.import_path)
            if isinstance(implementation, SubagentSpec):
                subagent_entries.append((name, implementation))
            else:
                function_tools[name] = implementation

        tools: list[Any] = list(function_tools.values())

        # add subagents to tool catalog. as_subagent_tool handles non-negotiables that all subagents must have
        for name, spec in subagent_entries:
            tools.append(
                as_subagent_tool(
                    spec.agent,
                    name=name,
                    session_manager=session_manager,
                    hitl_tools=self._config.hitl_tools,
                    description=spec.agent.description,
                )
            )

        # add MCP to tool catalog; lazily loads and connects to MCP clients as needed
        mcp_clients = await self._connect_mcp(context)
        tools.extend(mcp_clients)

        # define what tools require HITL
        hitl = self._hitl_intervention()
        interventions = [hitl] if hitl is not None else None

        logger.info(
            "agent created",
            extra={
                "subject_id": context.identity.subject_id,
                "tool_count": len(tools),
                "hitl_tools": sorted(self._config.hitl_tools),
                "mcp_hitl_tools": sorted(self._config.mcp_hitl_tools),
            },
        )

        return Agent(
            agent_id=ROOT_AGENT_ID,
            name=self._config.agent.name,
            system_prompt=self._config.agent.instructions,
            model=self._model,
            tools=tools,
            session_manager=session_manager,
            interventions=interventions,
            trace_attributes=create_trace_attributes(context),
            hooks=[GLOBAL_DEADLINE_HOOK],
            callback_handler=None,
        )
