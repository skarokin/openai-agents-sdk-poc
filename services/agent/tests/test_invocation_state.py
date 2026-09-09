"""Invocation-state contract tests.

Update EXPECTED_INVOCATION_STATE_KEYS when AgentContext.to_invocation_state grows.
"""

import threading

from conftest import make_context

from agent.core.models import IdentityContext

# Keep in sync with AgentContext.to_invocation_state().
EXPECTED_INVOCATION_STATE_KEYS = frozenset(
    {
        "token_vault",
        "identity",
        "request_id",
        "session_id",
        "soft_deadline_epoch_seconds",
        "hard_deadline_epoch_seconds",
        "cancel_signal",
    }
)


def test_invocation_state_contains_expected_payload():
    context = make_context(
        subject_id="alice",
        roles={"calculator"},
        request_id="req-1",
        session_id="sess-1",
    )
    state = context.to_invocation_state()

    assert frozenset(state) == EXPECTED_INVOCATION_STATE_KEYS
    assert state["identity"] is context.identity
    assert isinstance(state["identity"], IdentityContext)
    assert state["identity"].subject_id == "alice"
    assert state["identity"].roles == frozenset({"calculator"})
    assert state["request_id"] == "req-1"
    assert state["session_id"] == "sess-1"
    assert state["token_vault"] is context.token_vault
    assert state["soft_deadline_epoch_seconds"] == context.soft_deadline_epoch_seconds
    assert state["hard_deadline_epoch_seconds"] == context.hard_deadline_epoch_seconds
    assert state["cancel_signal"] is context.cancel_signal
    assert isinstance(state["cancel_signal"], threading.Event)
