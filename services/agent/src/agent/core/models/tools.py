"""Tool and subagent definitions."""

from dataclasses import dataclass

from strands import Agent


@dataclass(frozen=True, slots=True)
class SubagentSpec:
    """Catalog entry wrapping a fully defined nested Agent (YAML auto-discovery)."""

    agent: Agent
