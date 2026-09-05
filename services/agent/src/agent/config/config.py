"""Typed configuration loaded from the service YAML files."""

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentConfig(ConfigModel):
    name: str
    model: str
    instructions: str
    max_turns: int = Field(default=10, ge=1)


class McpToolConfig(ConfigModel):
    allowed_roles: frozenset[str] = Field(default_factory=frozenset)
    hitl: bool = False


class ToolConfig(ConfigModel):
    import_path: str
    type: Literal["function", "mcp"] = "function"
    enabled: bool = True
    allowed_roles: frozenset[str] = Field(default_factory=frozenset)
    hitl: bool = False
    tools: dict[str, McpToolConfig] = Field(default_factory=dict)


class ServiceConfig(ConfigModel):
    agent: AgentConfig
    tools: dict[str, ToolConfig]

    @property
    def hitl_tools(self) -> frozenset[str]:
        """Function / subagent catalog entries that require HumanInTheLoop."""

        return frozenset(
            name
            for name, settings in self.tools.items()
            if settings.type != "mcp" and settings.enabled and settings.hitl
        )

    @property
    def mcp_hitl_tools(self) -> frozenset[str]:
        """MCP child tool names that require HumanInTheLoop."""

        names: set[str] = set()
        for settings in self.tools.values():
            if settings.type != "mcp" or not settings.enabled:
                continue
            for child_name, child in settings.tools.items():
                if child.hitl:
                    names.add(child_name)
        return frozenset(names)

    @property
    def all_hitl_tools(self) -> frozenset[str]:
        return self.hitl_tools | self.mcp_hitl_tools


def hitl_allowed_tools(required: frozenset[str] | set[str]) -> list[str]:
    return ["*", *[f"!{name}" for name in sorted(required)]]


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected an object in {path}")
    return value


def load_service_config(config_dir: Path | None = None) -> ServiceConfig:
    directory = config_dir or Path(__file__).resolve().parent
    agent = AgentConfig.model_validate(_load_yaml(directory / "agent_config.yaml"))
    configured_model = os.getenv("AGENT_MODEL")
    if configured_model:
        agent = agent.model_copy(update={"model": configured_model})
    tools = {
        name: ToolConfig.model_validate(settings)
        for name, settings in _load_yaml(directory / "tool_config.yaml").items()
    }
    return ServiceConfig(agent=agent, tools=tools)
