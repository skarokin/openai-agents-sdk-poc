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
from agent.controls.interrupts import nested_hitl_reason
from agent.core.event_mapping import final_output_text, tool_use_from_message
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
    tool_use_message: Any = None,
) -> Any:
    """
    Tag bubbled nested interrupts so UI can show the subagent, not the root.

    Native HITL keeps a string reason; we only wrap it so the root can carry
    agent_name plus tool name/args from the nested interrupt's tool_use_message.
    """

    reason = interrupt.reason
    if isinstance(reason, dict) and reason.get("requires_auth"):
        return {**reason, "agent_name": agent_name}

    tool_name, arguments = tool_use_from_message(tool_use_message, interrupt.id)
    return nested_hitl_reason(
        agent_name=agent_name,
        tool_name=tool_name,
        arguments=arguments,
    )


def _subagent_tool(
    spec: SubagentSpec,
    *,
    name: str,
    tools: list[Any],
    model: str | Model,
    hitl_tools: frozenset[str] | set[str],
    session_manager: FileSessionManager,
) -> Any:
    """
    Build a nested agent-as-tool that shares the parent conversation session.

    An agent-as-tool...
    1. Does not inherit the parent's invocation_state
    2. Does not natively bubble up interrupts to the parent agent
    3. (this list will grow as we add more features that need to be handled)

    So, we ensure that:
    1. Invocation_state is passed in to the nested agent via ToolContext
    2. Interrupts made on the subagent bubble up to the root agent
    3. Interrupts made in response to subagents are bubbled back down to the subagent.

    The session manager being shared is critical for the interrupt state to be shared between root and subagents.
    """

    nested_hitl = frozenset(hitl_tools) & frozenset(spec.tools)
    display_name = spec.agent_name or name
    nested = Agent(
        agent_id=name,
        name=display_name,
        description=spec.description,
        system_prompt=spec.instructions,
        model=_openai_model(model),
        tools=tools,
        session_manager=session_manager,
        interventions=(
            [HumanInTheLoop(allowed_tools=hitl_allowed_tools(nested_hitl))]
            if nested_hitl
            else None
        ),
        callback_handler=None,
    )

    async def run(input: str, tool_context: ToolContext) -> str:
        invocation_state = {
            key: value
            for key, value in tool_context.invocation_state.items()
            if key != "agent"
        }

        if nested._interrupt_state.activated:
            responses = []
            tool_use_message = nested._interrupt_state.context.get("tool_use_message")
            for interrupt in nested._interrupt_state.interrupts.values():
                response = tool_context.interrupt(
                    interrupt.id,
                    reason=_nested_interrupt_reason(
                        interrupt,
                        agent_name=display_name,
                        tool_use_message=tool_use_message,
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
            result = await nested.invoke_async(
                responses, invocation_state=invocation_state
            )
        else:
            result = await nested.invoke_async(
                input, invocation_state=invocation_state
            )

        if result.stop_reason == "interrupt" and result.interrupts:
            tool_use_message = nested._interrupt_state.context.get("tool_use_message")
            for interrupt in result.interrupts:
                tool_context.interrupt(
                    interrupt.id,
                    reason=_nested_interrupt_reason(
                        interrupt,
                        agent_name=display_name,
                        tool_use_message=tool_use_message,
                    ),
                )
            raise RuntimeError("nested interrupt should have raised")

        return final_output_text(result.message) or str(result)

    decorated = tool(name=name, description=spec.description, context=True)(run)
    decorated._nested_agent = nested
    return decorated


def _openai_model(model: str | Model) -> Model:
    if isinstance(model, Model):
        return model

    return OpenAIModel(
        model_id=model,
        params={
            "reasoning_effort": "none"
        }
    )


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
        self._model = _openai_model(model if model is not None else config.agent.model)

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
        subagent_entries: list[tuple[str, ToolConfig, SubagentSpec]] = []

        for name, settings in self._config.tools.items():
            if not settings.enabled or settings.type == "mcp":
                continue
            if not _has_role(context, settings.allowed_roles):
                continue

            implementation = _load_symbol(settings.import_path)
            if isinstance(implementation, SubagentSpec):
                subagent_entries.append((name, settings, implementation))
                continue

            function_tools[name] = implementation

        tools: list[Any] = list(function_tools.values())

        for name, _settings, spec in subagent_entries:
            leaf_tools = [
                function_tools[tool_name]
                for tool_name in spec.tools
                if tool_name in function_tools
            ]
            if len(leaf_tools) < len(spec.tools):
                for tool_name in spec.tools:
                    if tool_name in function_tools:
                        continue
                    tool_settings = self._config.tools.get(tool_name)
                    if tool_settings is None or tool_settings.type == "mcp":
                        continue
                    leaf_tools.append(_load_symbol(tool_settings.import_path))

            tools.append(
                _subagent_tool(
                    spec,
                    name=name,
                    tools=leaf_tools,
                    model=self._model,
                    hitl_tools=self._config.hitl_tools,
                    session_manager=session_manager,
                )
            )

        mcp_clients = await self._connect_mcp(context)
        tools.extend(mcp_clients)

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
            state=context.to_agent_state(),
            trace_attributes=create_trace_attributes(context),
            callback_handler=None,
        )
