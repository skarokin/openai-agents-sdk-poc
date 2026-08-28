"""
core models for the agent service
"""
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol

from agents import RunState, TResponseInputItem


@dataclass(frozen=True, slots=True)
class IdentityContext:
    """
    The effective identity of the caller - assumes authn and authz are handled out-of-band.

    Supports delegation/OBO flows via the actor_id field
    """

    subject_id: str
    actor_id: str | None = None
    roles: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_delegated(self) -> bool:
        return self.actor_id is not None


@dataclass(frozen=True, slots=True)
class Query:
    """Input for either a new agent run or a resumed interrupted run."""

    input: str | list[TResponseInputItem] | RunState[Any]

    @property
    def is_resume(self) -> bool:
        return isinstance(self.input, RunState)


@dataclass(frozen=True, slots=True)
class UsageUpdate:
    """Cumulative usage observed during the run."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """A tool invocation awaiting an external approval decision."""

    interruption_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    agent_name: str | None = None


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
class GuardrailTripped:
    """A guardrail blocked the run or a tool invocation."""

    guardrail_name: str
    kind: Literal["input", "output", "tool_input", "tool_output"]
    message: str | None = None


@dataclass(frozen=True, slots=True)
class AgentChanged:
    """Execution was handed off to another agent."""

    agent_name: str
    previous_agent_name: str | None = None


@dataclass(frozen=True, slots=True)
class TurnComplete:
    """The run completed successfully."""

    output: Any
    usage: UsageUpdate


@dataclass(frozen=True, slots=True)
class TurnInterrupt:
    """The run paused pending one or more approval decisions."""

    interruptions: tuple[ApprovalRequest, ...]
    run_state: RunState[Any]


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
    | GuardrailTripped
    | UsageUpdate
    | AgentChanged
    | TurnOutcome
)


@dataclass(frozen=True, slots=True)
class EventSource:
    """Source agent and invocation of an event in the run"""

    agent_name: str
    invocation_id: str
    parent_invocation_id: str | None = None
    kind: Literal["root", "handoff", "agent_tool", "a2a"] = "root"


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    """Envelope for an event in the run"""

    sequence: int
    source: EventSource
    event: RunEvent


class EventSink(Protocol):
    """Destination for normalized events produced by nested work."""

    async def emit(self, event: RunEvent, *, source: EventSource) -> None:
        """Publish an event for inclusion in the formatter's output stream."""
        ...


class NoOpEventSink:
    """Discard events when the caller does not request streaming output."""

    async def emit(self, event: RunEvent, *, source: EventSource) -> None:
        pass


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Per-run dependencies and identity available to agents and tools."""

    identity: IdentityContext
    event_sink: EventSink


@dataclass(frozen=True, slots=True)
class CompleteResult:
    """Result of an agent turn using the complete() formatter"""

    outcome: TurnOutcome
