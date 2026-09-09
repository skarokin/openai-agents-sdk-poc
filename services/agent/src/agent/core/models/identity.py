"""Caller identity carried through a run."""

from __future__ import annotations

from dataclasses import dataclass, field

from strands.types.tools import ToolContext


@dataclass(frozen=True, slots=True)
class IdentityContext:
    """
    The effective identity of the caller.

    Assumes authn and authz are handled out-of-band. Supports delegation/OBO
    flows via the actor_id field.
    """

    subject_id: str
    actor_id: str | None = None
    roles: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_delegated(self) -> bool:
        return self.actor_id is not None

    @classmethod
    def from_tool_context(cls, tool_context: ToolContext) -> IdentityContext:
        """Resolve identity from invocation_state (required)."""

        invocation_identity = tool_context.invocation_state.get("identity")
        if isinstance(invocation_identity, IdentityContext):
            return invocation_identity

        raise RuntimeError(
            "invocation_state missing IdentityContext under 'identity'"
        )
