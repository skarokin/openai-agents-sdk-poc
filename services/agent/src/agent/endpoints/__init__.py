"""
Caller-facing entry points into the agent (HTTP, MCP, A2A, ...).

Endpoints:
- expose something a caller can hit
- parse the request
- read trusted identity
- call AgentRunner (sync_run/stream_run/sync_resume/stream_resume)
- map the runner result back to the caller's protocol

Endpoints never construct or invoke the Strands Agent directly.
"""

from .http_rest import router as rest_router
from .http_sse import router as sse_router

__all__ = ["rest_router", "sse_router"]
