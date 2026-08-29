"""Domain models for the agent service."""

from dataclasses import dataclass

from .context import AgentContext, Query
from .events import (
    AgentChanged,
    AuthRequired,
    EventSource,
    GuardrailTripped,
    ReasoningChunk,
    TextChunk,
    ToolResult,
    ToolStart,
    UsageUpdate,
    ApprovalRequest,
    CompleteResult,
    TurnComplete,
    TurnError,
    TurnInterrupt,
    TurnOutcome,
)
from .identity import IdentityContext
from .sink import EventSink
from .tools import SubagentSpec

type RunEvent = (
    TextChunk
    | ReasoningChunk
    | ToolStart
    | ToolResult
    | GuardrailTripped
    | AuthRequired
    | UsageUpdate
    | AgentChanged
    | TurnOutcome
    | ApprovalRequest
)


class NoOpEventSink:
    """Discard events when the caller does not request streaming output."""

    async def emit(self, event: RunEvent, *, source: EventSource) -> None:
        return None


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
    "EventSink",
    "EventSource",
    "GuardrailTripped",
    "IdentityContext",
    "NoOpEventSink",
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
