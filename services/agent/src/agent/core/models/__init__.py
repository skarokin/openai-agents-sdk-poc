"""Domain models for the agent service."""

from dataclasses import dataclass

from .context import AgentContext, Query
from .events import (
    AgentChanged,
    ApprovalRequest,
    AuthRequired,
    CompleteResult,
    EventSource,
    ReasoningChunk,
    TextChunk,
    ToolResult,
    ToolStart,
    TurnComplete,
    TurnError,
    TurnInterrupt,
    TurnOutcome,
    UsageUpdate,
)
from .identity import IdentityContext
from .tools import SubagentSpec

type RunEvent = (
    TextChunk
    | ReasoningChunk
    | ToolStart
    | ToolResult
    | AuthRequired
    | UsageUpdate
    | AgentChanged
    | TurnOutcome
    | ApprovalRequest
)


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    """Envelope for an event in the run."""

    sequence: int
    source: EventSource
    event: RunEvent


__all__ = [
    "AgentChanged",
    "AgentContext",
    "ApprovalRequest",
    "AuthRequired",
    "CompleteResult",
    "EventEnvelope",
    "EventSource",
    "IdentityContext",
    "Query",
    "ReasoningChunk",
    "RunEvent",
    "SubagentSpec",
    "TextChunk",
    "ToolResult",
    "ToolStart",
    "TurnComplete",
    "TurnError",
    "TurnInterrupt",
    "TurnOutcome",
    "UsageUpdate",
]
