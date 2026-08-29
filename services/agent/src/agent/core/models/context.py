"""Per-run execution context."""

import time
from dataclasses import dataclass, field
from typing import Any

from agents import RunState, TResponseInputItem

from .identity import IdentityContext
from .sink import EventSink


@dataclass(frozen=True, slots=True)
class Query:
    """Input for either a new agent run or a resumed interrupted run."""

    input: str | list[TResponseInputItem] | RunState[Any]

    @property
    def is_resume(self) -> bool:
        return isinstance(self.input, RunState)


def _default_token_vault():
    from agent.core.auth.token_vault import TokenVault

    return TokenVault.from_environment()


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Per-run dependencies and identity available to agents and tools."""

    identity: IdentityContext
    event_sink: EventSink
    request_id: str
    session_id: str
    deadline_epoch_seconds: float
    token_vault: Any = field(default_factory=_default_token_vault)

    @property
    def deadline_exceeded(self) -> bool:
        return time.time() >= self.deadline_epoch_seconds
