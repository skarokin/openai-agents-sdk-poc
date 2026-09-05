"""Per-run execution context."""

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .identity import IdentityContext


@dataclass(frozen=True, slots=True)
class Query:
    """Input for either a new agent run or a resumed interrupted run."""

    input: str | list[dict[str, Any]]

    @property
    def is_resume(self) -> bool:
        return isinstance(self.input, list)


def _default_token_vault():
    from agent.core.token_vault import TokenVault

    return TokenVault.from_environment()


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Per-run identity/deps via invocation_state and agent.state."""

    identity: IdentityContext
    request_id: str
    session_id: str
    soft_deadline_epoch_seconds: float
    hard_deadline_epoch_seconds: float
    token_vault: Any = field(default_factory=_default_token_vault)
    cancel_signal: threading.Event = field(default_factory=threading.Event)

    @property
    def soft_deadline_exceeded(self) -> bool:
        return time.time() >= self.soft_deadline_epoch_seconds

    @property
    def hard_deadline_exceeded(self) -> bool:
        return time.time() >= self.hard_deadline_epoch_seconds

    def to_agent_state(self) -> dict[str, Any]:
        """JSON-serializable identity/session facts stored on agent.state."""

        return {
            "identity": {
                "subject_id": self.identity.subject_id,
                "actor_id": self.identity.actor_id,
                "roles": sorted(self.identity.roles),
            },
            "request_id": self.request_id,
            "session_id": self.session_id,
            "soft_deadline_epoch_seconds": self.soft_deadline_epoch_seconds,
            "hard_deadline_epoch_seconds": self.hard_deadline_epoch_seconds,
        }

    def to_invocation_state(self) -> dict[str, Any]:
        """Per-invocation deps (not persisted) passed into invoke/stream."""

        return {
            "token_vault": self.token_vault,
            "identity": self.identity,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "soft_deadline_epoch_seconds": self.soft_deadline_epoch_seconds,
            "hard_deadline_epoch_seconds": self.hard_deadline_epoch_seconds,
            "cancel_signal": self.cancel_signal,
        }
