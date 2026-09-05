"""Create per-request agents from global tool config and caller identity."""

import asyncio
import importlib
import logging
import re
from typing import Any

from strands import Agent, tool
from strands.models import Model
from strands.models.openai import OpenAIModel
from strands.tools.mcp import MCPClient
from strands.types.tools import ToolContext
from strands.vended_interventions.hitl import HumanInTheLoop

from agent.config import ServiceConfig, ToolConfig, hitl_allowed_tools
from agent.core.event_mapping import final_output_text
from agent.core.models import AgentContext, SubagentSpec
from agent.core.runtime import create_trace_attributes
from agent.core.sessions import make_file_session_manager

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


def _subagent_tool(
    spec: SubagentSpec,
    *,
    name: str,
    tools: list[Any],
    model: str | Model,
    hitl_tools: frozenset[str] | set[str],
) -> Any:
    """
    Build an agent-as-tool that injects any required dependencies from the root agent.

    An agent-as-tool does not inherit the parent's invocation_state; but we CAN access invocation state
    via ToolContext and *then* pass it in to the nested agent.

    This also handles bubbling up/down interrupts to the root agent.
    """

    nested_hitl = frozenset(hitl_tools) & frozenset(spec.tools)
    nested = Agent(
        name=name,
        description=spec.description,
        system_prompt=spec.instructions,
        model=_openai_model(model),
        tools=tools,
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
            if key != "agent"   # drop the parent agent itself
        }

        # invoke subagent - bubble down interrupt if any else regular invoke
        if nested._interrupt_state.activated:
            responses = []
            for interrupt in nested._interrupt_state.interrupts.values():
                response = tool_context.interrupt(
                    interrupt.id,
                    reason=interrupt.reason,
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

        # bubble up interrupt to the root agent
        if result.stop_reason == "interrupt" and result.interrupts:
            for interrupt in result.interrupts:
                tool_context.interrupt(interrupt.id, reason=interrupt.reason)

            raise RuntimeError("nested interrupt should have raised")

        return final_output_text(result.message) or str(result)

    return tool(name=name, description=spec.description, context=True)(run)


def _openai_model(model: str | Model) -> Model:
    if isinstance(model, Model):
        return model

    return OpenAIModel(
        model_id=model,
        params={
            "reasoning_effort": "none"
        }
    )


def _agent_cache_key(context: AgentContext) -> str:
    roles = ",".join(sorted(context.identity.roles))
    return f"{context.identity.subject_id}:{context.session_id}:{roles}"


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
        self._agents: dict[str, Agent] = {}
        self._agent_lock = asyncio.Lock()
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
        async with self._agent_lock:
            self._agents.clear()
        async with self._mcp_lock:
            for client in self._mcp_clients.values():
                client.stop(None, None, None)
            self._mcp_clients.clear()

    def _hitl_intervention(self) -> HumanInTheLoop | None:
        required = self._config.all_hitl_tools
        if not required:
            return None
        return HumanInTheLoop(allowed_tools=hitl_allowed_tools(required))

    def _refresh_cached_agent(self, agent: Agent, context: AgentContext) -> Agent:
        """Apply this request's identity/session facts onto a reused Agent.

        Tools and nested agents stay as-is; only ``agent.state`` keys from
        ``context.to_agent_state()`` are updated for the new request_id /
        deadline / identity snapshot.
        """

        for key, value in context.to_agent_state().items():
            agent.state.set(key, value)
        return agent

    async def create(self, context: AgentContext) -> Agent:
        # Reuse the same Agent tree across run/resume so nested interrupt state
        # is not wiped by rebuilding tools each HTTP request.
        cache_key = _agent_cache_key(context)
        async with self._agent_lock:
            cached = self._agents.get(cache_key)
            if cached is not None:
                return self._refresh_cached_agent(cached, context)

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

            # as_tool() drops invocation_state; custom wrap forwards vault/identity.
            tools.append(
                _subagent_tool(
                    spec,
                    name=name,
                    tools=leaf_tools,
                    model=self._model,
                    hitl_tools=self._config.hitl_tools,
                )
            )

        mcp_clients = await self._connect_mcp(context)
        tools.extend(mcp_clients)

        session_manager = make_file_session_manager(
            context.session_id,
            context.identity.subject_id,
        )

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

        agent = Agent(
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
        async with self._agent_lock:
            existing = self._agents.get(cache_key)
            if existing is not None:
                return self._refresh_cached_agent(existing, context)
            self._agents[cache_key] = agent
        return agent
