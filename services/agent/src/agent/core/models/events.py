"""Streaming and lifecycle events emitted during a run."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class UsageUpdate:
    """Cumulative usage observed during the run."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class TextChunk:
    """An incremental final-answer text fragment."""

    delta: str


@dataclass(frozen=True, slots=True)
class ReasoningChunk:
    """An incremental reasoning summary fragment, when available."""

    delta: str


@dataclass(frozen=True, slots=True)
class ToolStart:
    """A tool invocation has started."""

    call_id: str
    tool_name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResult:
    """A tool invocation has completed."""

    call_id: str
    tool_name: str
    output: Any


@dataclass(frozen=True, slots=True)
class SubagentEvent:
    """Tool activity from inside a nested agent-as-tool (not text/reasoning)."""

    agent_name: str
    event: ToolStart | ToolResult


@dataclass(frozen=True, slots=True)
class AuthRequired:
    """A tool was blocked because the caller lacks a vault token for a service."""

    call_id: str
    tool_name: str
    service: str
    authorization_url: str


@dataclass(frozen=True, slots=True)
class GuardrailTripped:
    """A guardrail blocked or flagged content during the run."""

    message: str
    name: str | None = None
    details: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class Annotation:
    """Out-of-band enrichment on a run (citations, grounding, provenance, etc.)."""

    kind: str
    data: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class AgentChanged:
    """Execution was handed off to another agent."""

    agent_name: str
    previous_agent_name: str | None = None


@dataclass(frozen=True, slots=True)
class EventSource:
    """Source agent and invocation of an event in the run."""

    agent_name: str
    invocation_id: str
    parent_invocation_id: str | None = None
    kind: Literal["root", "handoff", "agent_tool", "a2a", "mcp"] = "root"


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """A tool invocation awaiting an external approval / auth decision."""

    interruption_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    agent_name: str | None = None
    requires_auth: bool = False
    requires_approval: bool = True
    authenticated: bool = False
    service: str | None = None
    authorization_url: str | None = None


@dataclass(frozen=True, slots=True)
class TurnComplete:
    """The run completed successfully."""

    output: Any
    usage: UsageUpdate
    auth_required: tuple[AuthRequired, ...] = ()


@dataclass(frozen=True, slots=True)
class TurnInterrupt:
    """The run paused pending interrupt responses (session holds the state)."""

    interruptions: tuple[ApprovalRequest, ...]


@dataclass(frozen=True, slots=True)
class TurnError:
    """The run failed with a protocol-safe error."""

    code: str
    message: str
    retryable: bool = False


type TurnOutcome = TurnComplete | TurnInterrupt | TurnError

type RunEvent = (
    TextChunk
    | ReasoningChunk
    | ToolStart
    | ToolResult
    | SubagentEvent
    | AuthRequired
    | GuardrailTripped
    | Annotation
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


@dataclass(frozen=True, slots=True)
class CompleteResult:
    """Result of an agent turn collected from the streaming runner."""

    outcome: TurnOutcome
