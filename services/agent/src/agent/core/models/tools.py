"""Tool and subagent definitions."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SubagentSpec:
    """Definition of an agent-as-tool. Instances can live in any module."""

    agent_name: str
    instructions: str
    description: str
    tools: tuple[str, ...]
