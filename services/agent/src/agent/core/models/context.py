"""Per-run execution context."""

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .identity import IdentityContext


def _default_token_vault():
    from agent.core.token_vault import TokenVault

    return TokenVault.from_environment()


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Per-run identity and deps carried via invocation_state."""

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
