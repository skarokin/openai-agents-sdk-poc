"""Caller identity carried through a run."""

from dataclasses import dataclass, field


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
