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


class ToolConfig(ConfigModel):
    import_path: str
    type: Literal["function", "mcp"] = "function"
    enabled: bool = True
    allowed_roles: frozenset[str] = Field(default_factory=frozenset)
    tools: dict[str, McpToolConfig] = Field(default_factory=dict)


class ServiceConfig(ConfigModel):
    agent: AgentConfig
    tools: dict[str, ToolConfig]


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected an object in {path}")
    return value


def load_service_config(config_dir: Path | None = None) -> ServiceConfig:
    directory = config_dir or Path(__file__).resolve().parent.parent / "config"
    agent = AgentConfig.model_validate(_load_yaml(directory / "agent_config.yaml"))
    configured_model = os.getenv("AGENT_MODEL")
    if configured_model:
        agent = agent.model_copy(update={"model": configured_model})
    tools = {
        name: ToolConfig.model_validate(settings)
        for name, settings in _load_yaml(directory / "tool_config.yaml").items()
    }
    return ServiceConfig(agent=agent, tools=tools)
