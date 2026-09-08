"""
Runs the agent and returns results for endpoints (HTTP, MCP, A2A, ...).

One streaming turn path; sync callers collect the terminal event.
"""

from .agent import AgentRunner

__all__ = ["AgentRunner"]
