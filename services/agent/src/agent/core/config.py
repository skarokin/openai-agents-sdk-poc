"""Typed configuration loaded from the service YAML files."""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentConfig(ConfigModel):
    name: str
    model: str
    instructions: str
    max_turns: int = Field(default=10, ge=1)


class ToolConfig(ConfigModel):
    enabled: bool = True
    allowed_roles: frozenset[str] = Field(default_factory=frozenset)
    requires_approval: bool = False


class ToolRegistryConfig(ConfigModel):
    tools: dict[str, ToolConfig]


class ServiceConfig(ConfigModel):
    agent: AgentConfig
    tools: ToolRegistryConfig


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
    tools = ToolRegistryConfig.model_validate(
        _load_yaml(directory / "tool_config.yaml")
    )
    return ServiceConfig(agent=agent, tools=tools)
