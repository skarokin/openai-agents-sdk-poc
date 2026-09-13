"""Auth interrupt helpers and vault behavior."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import make_context
from strands.interrupt import Interrupt, InterruptException

from agent.controls.auth import require_vault_token
from agent.core.event_mapping import map_interrupts
from agent.core.models import IdentityContext
from agent.core.token_vault import TokenVault


@pytest.mark.asyncio
async def test_vault_put_get_round_trip():
    with tempfile.TemporaryDirectory() as directory:
        vault = TokenVault(Path(directory))
        identity = IdentityContext(subject_id="user-1", roles=frozenset({"auth_user"}))
        assert await vault.has_token(identity, "authentication_demo") is False
        await vault.put(identity, "authentication_demo", "tok-123")
        assert await vault.has_token(identity, "authentication_demo") is True
        stored = await vault.get(identity, "authentication_demo")
        assert stored is not None
        assert stored.access_token == "tok-123"


def test_vault_challenge_url():
    vault = TokenVault(Path(tempfile.mkdtemp()))
    challenge = vault.challenge("authentication_demo")
    assert challenge.service == "authentication_demo"
    assert challenge.authorization_url == "agent://vault/authentication_demo"


@pytest.mark.asyncio
async def test_auth_interrupt_from_within_nested_agent_context(tmp_path):
    """Auth interrupt is raised by the tool body, independent of HITL."""

    vault = TokenVault(tmp_path)
    identity = make_context(roles={"auth_user"}, token_vault=vault).identity
    nested_agent = SimpleNamespace(name="Open Subagent")

    def interrupt(name, reason=None):
        raise InterruptException(
            Interrupt(id="auth-nested-1", name=name, reason=reason)
        )

    tool_context = SimpleNamespace(
        agent=nested_agent,
        invocation_state={"token_vault": vault, "identity": identity},
        interrupt=interrupt,
    )

    with pytest.raises(InterruptException) as raised:
        await require_vault_token(
            tool_context,  # type: ignore[arg-type]
            service="authentication_demo",
            tool_name="authentication_demo",
        )

    interrupt_obj = raised.value.interrupt
    assert interrupt_obj.name == "auth-authentication_demo"
    assert interrupt_obj.reason["requires_auth"] is True
    assert interrupt_obj.reason["requires_approval"] is False

    mapped = map_interrupts(
        [interrupt_obj],
        agent_context=make_context(roles={"auth_user"}, token_vault=vault),
        agent_name="Open Subagent",
    )
    assert mapped[0].requires_auth is True
    assert mapped[0].requires_approval is False
    assert mapped[0].agent_name == "Open Subagent"


def test_auth_and_hitl_are_separate_interrupt_shapes():
    auth = Interrupt(
        id="a1",
        name="auth-authentication_demo",
        reason={
            "requires_auth": True,
            "requires_approval": False,
            "service": "authentication_demo",
            "authorization_url": "agent://vault/authentication_demo",
            "tool_name": "authentication_demo",
            "arguments": {},
        },
    )
    hitl = Interrupt(
        id="v1:before_tool_call:call-hitl:deadbeef",
        name="strands:human-in-the-loop",
        reason='Approve "auth_approval_demo"?\n  Input: {}',
    )
    mapped = map_interrupts(
        [hitl, auth],
        agent_name="root",
    )
    assert mapped[0].requires_approval is True and mapped[0].requires_auth is False
    assert mapped[0].tool_name == "auth_approval_demo"
    assert mapped[1].requires_auth is True and mapped[1].requires_approval is False


def test_identity_from_tool_context_requires_invocation_state():
    identity = IdentityContext(subject_id="user-1", roles=frozenset({"auth_user"}))
    ok = SimpleNamespace(invocation_state={"identity": identity})
    assert IdentityContext.from_tool_context(ok) is identity  # type: ignore[arg-type]

    missing = SimpleNamespace(invocation_state={})
    with pytest.raises(RuntimeError, match="invocation_state missing IdentityContext"):
        IdentityContext.from_tool_context(missing)  # type: ignore[arg-type]
