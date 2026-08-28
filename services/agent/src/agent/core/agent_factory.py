"""Create per-request agents from global tool config and caller identity."""

import importlib
import logging
from typing import Any

from agents import Agent, FunctionTool

from agent.common.binding import bind_function_tool, bind_subagent_tool

from .config import ServiceConfig, ToolConfig
from .models import AgentContext, SubagentSpec

logger = logging.getLogger(__name__)


def _load_symbol(import_path: str) -> Any:
    module_name, separator, attr = import_path.partition(":")
    if not separator or not attr:
        raise ValueError(f"Tool import_path must be 'module:attr', got {import_path}")
    return getattr(importlib.import_module(module_name), attr)


class AgentFactory:
    """Register configured tools; identity decides which ones are attached."""

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

    def create(self, context: AgentContext) -> Agent[AgentContext]:
        function_tools: dict[str, FunctionTool] = {}
        subagent_entries: list[tuple[str, ToolConfig, SubagentSpec]] = []

        for name, settings in self._config.tools.items():
            if not self._authorized(settings, context):
                continue
            implementation = _load_symbol(settings.import_path)
            if isinstance(implementation, SubagentSpec):
                subagent_entries.append((name, settings, implementation))
                continue
            function_tools[name] = bind_function_tool(
                implementation,
                name=name,
                needs_approval=settings.requires_approval,
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
                    needs_approval=settings.requires_approval,
                )
            )

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
