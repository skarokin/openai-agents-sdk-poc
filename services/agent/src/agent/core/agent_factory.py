"""Create per-request agents from global tool config and caller identity."""

import asyncio
import importlib
import logging
from typing import Any

from agents import Agent, FunctionTool
from agents.mcp import MCPServer

from agent.common.binding import bind_function_tool, bind_subagent_tool
from agent.config import ServiceConfig, ToolConfig
from agent.core.models import AgentContext, SubagentSpec


logger = logging.getLogger(__name__)


def _load_symbol(import_path: str) -> Any:
    module_name, separator, attr = import_path.partition(":")
    if not separator or not attr:
        raise ValueError(f"Tool import_path must be 'module:attr', got {import_path}")

    return getattr(importlib.import_module(module_name), attr)


def _has_role(context: AgentContext, allowed_roles: frozenset[str]) -> bool:
    return bool(context.identity.roles & allowed_roles)


def _mcp_tool_filter(settings: ToolConfig):
    def is_allowed(filter_ctx, tool) -> bool:
        roles = filter_ctx.run_context.context.identity.roles
        if not (roles & settings.allowed_roles):
            return False
        child = settings.tools.get(tool.name)
        if child is None or not child.allowed_roles:
            return True
        return bool(roles & child.allowed_roles)

    return is_allowed


class AgentFactory:
    """Returns an agent with the catalog attached. `is_enabled` hides tools the caller cannot use."""

    def __init__(self, config: ServiceConfig, model: Any | None = None):
        self._config = config
        self._model = model or config.agent.model
        self._mcp_servers: dict[str, MCPServer] = {}
        self._mcp_lock = asyncio.Lock()

    @property
    def max_turns(self) -> int:
        return self._config.agent.max_turns

    async def _connect_mcp(self, context: AgentContext) -> list[MCPServer]:
        """Startup & connect to all MCP servers the caller is allowed to use"""
        wanted = [
            (name, settings)
            for name, settings in self._config.tools.items()
            if settings.type == "mcp"
            and settings.enabled
            and _has_role(context, settings.allowed_roles)
        ]
        servers: list[MCPServer] = []

        async with self._mcp_lock:
            for name, settings in wanted:
                server = self._mcp_servers.get(name)
                if server is None:
                    create = _load_symbol(settings.import_path)
                    server = create(tool_filter=_mcp_tool_filter(settings))
                    await server.connect()
                    self._mcp_servers[name] = server
                servers.append(server)

        return servers

    async def close(self) -> None:
        """Cleanup factory-made resources"""

        # cleanup MCP servers
        async with self._mcp_lock:
            for server in self._mcp_servers.values():
                await server.cleanup()

            self._mcp_servers.clear()

        # ... other cleanup when needed

    async def create(self, context: AgentContext) -> Agent[AgentContext]:
        function_tools: dict[str, FunctionTool] = {}
        subagent_entries: list[tuple[str, ToolConfig, SubagentSpec]] = []

        for name, settings in self._config.tools.items():
            if not settings.enabled or settings.type == "mcp":
                continue

            implementation = _load_symbol(settings.import_path)
            if isinstance(implementation, SubagentSpec):
                subagent_entries.append((name, settings, implementation))
                continue

            function_tools[name] = bind_function_tool(
                implementation,
                name=name,
                is_enabled=_has_role(context, settings.allowed_roles),
            )

        tools: list[Any] = list(function_tools.values())

        for name, settings, spec in subagent_entries:
            tools.append(
                bind_subagent_tool(
                    spec,
                    name=name,
                    tools=[
                        function_tools[tool_name]
                        for tool_name in spec.tools
                        if tool_name in function_tools
                    ],
                    model=self._model,
                    max_turns=self._config.agent.max_turns,
                    context=context,
                    is_enabled=_has_role(context, settings.allowed_roles),
                )
            )

        mcp_servers = await self._connect_mcp(context)

        logger.info(
            "agent created",
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
            mcp_servers=mcp_servers,
        )
