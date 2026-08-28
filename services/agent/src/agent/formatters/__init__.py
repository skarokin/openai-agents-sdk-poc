"""
runs the agent and returns the output in a different format. these are consumed by protocol adapters (e.g. HTTP, MCP, A2A) and returned to the caller.

there are currently two formatters:
- complete : runs to completion and returns one protocol-neutral response
- events : yields events (text chunks, reasoning chunks, tool calls, etc)

Use the complete formatter when only the final answer is needed.
Use the events formatter when incremental output, task progress, etc. is needed.
"""

from .complete import CompleteFormatter
from .events import EventsFormatter

__all__ = ["CompleteFormatter", "EventsFormatter"]
