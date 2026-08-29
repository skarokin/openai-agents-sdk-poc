"""Auth and HITL approval integration tests."""

import tempfile
from pathlib import Path

import pytest
from agents.testing import ScriptedModel, assistant_message, function_call
from conftest import make_context

from agent.core.agent_factory import AgentFactory
from agent.core.auth import AuthTokenMissingError, TokenVault, get_access_token
from agent.core.models import TurnComplete, TurnInterrupt
from agent.core.session_manager import SessionManager
from agent.formatters.complete import CompleteFormatter


@pytest.mark.asyncio
async def test_get_access_token_raises_when_vault_empty():
    vault = TokenVault(Path(tempfile.mkdtemp()))
    context = make_context(
        subject_id="user",
        roles={"auth_user"},
        request_id="req",
        session_id="session",
        deadline_offset_seconds=9_999_999_999.0,
        token_vault=vault,
    )

    with pytest.raises(AuthTokenMissingError) as raised:
        await get_access_token(context, "authentication_demo")

    assert raised.value.service == "authentication_demo"


@pytest.mark.asyncio
async def test_get_access_token_returns_token_when_present():
    with tempfile.TemporaryDirectory() as directory:
        vault = TokenVault(Path(directory))
        identity = make_context(
            subject_id="user",
            roles={"auth_user"},
        ).identity
        await vault.put(identity, "authentication_demo", "oauth-token")
        context = make_context(
            subject_id="user",
            roles={"auth_user"},
            request_id="req",
            session_id="session",
            deadline_offset_seconds=9_999_999_999.0,
            token_vault=vault,
        )

        token = await get_access_token(context, "authentication_demo")

        assert token == "oauth-token"


@pytest.mark.asyncio
async def test_auth_required_pauses_until_token_is_stored(service_config):
    with tempfile.TemporaryDirectory() as directory:
        data_dir = Path(directory)
        sessions = SessionManager(data_dir)
        vault = TokenVault(data_dir)
        model = ScriptedModel(
            [
                [function_call("authentication_demo", {}, call_id="call-auth")],
                [assistant_message("authenticated")],
            ]
        )
        factory = AgentFactory(service_config, model=model)
        formatter = CompleteFormatter(factory, sessions, vault)
        identity = make_context(
            subject_id="guardrail-user",
            roles={"auth_user"},
        ).identity
        try:
            first = await formatter.run(
                "Authenticate",
                identity,
                request_id="req-1",
                session_id="session-1",
            )
            assert isinstance(first.outcome, TurnInterrupt)
            challenge = first.outcome.interruptions[0]
            assert challenge.requires_auth is True
            assert challenge.requires_approval is False
            assert challenge.authenticated is False
            assert challenge.service == "authentication_demo"
            assert challenge.authorization_url == "agent://vault/authentication_demo"

            await vault.put(identity, "authentication_demo", "oauth-token")
            completed, _ = await formatter.resume(
                first.outcome.resume_token,
                {challenge.interruption_id: "approve"},
                identity,
                request_id="req-2",
            )
            assert isinstance(completed.outcome, TurnComplete)
            assert completed.outcome.output == "authenticated"
            model.assert_complete()
        finally:
            await factory.close()


@pytest.mark.asyncio
async def test_auth_required_runs_when_vault_has_token(service_config):
    with tempfile.TemporaryDirectory() as directory:
        data_dir = Path(directory)
        sessions = SessionManager(data_dir)
        vault = TokenVault(data_dir)
        identity = make_context(
            subject_id="guardrail-user",
            roles={"auth_user"},
        ).identity
        await vault.put(identity, "authentication_demo", "oauth-token")
        model = ScriptedModel(
            [
                [function_call("authentication_demo", {}, call_id="call-auth")],
                [assistant_message("authenticated")],
            ]
        )
        factory = AgentFactory(service_config, model=model)
        formatter = CompleteFormatter(factory, sessions, vault)
        try:
            result = await formatter.run(
                "Authenticate",
                identity,
                request_id="req-1",
                session_id="session-1",
            )
            assert isinstance(result.outcome, TurnComplete)
            assert result.outcome.output == "authenticated"
            model.assert_complete()
        finally:
            await factory.close()


@pytest.mark.asyncio
async def test_hitl_auth_required_single_interrupt_when_not_authenticated(
    service_config,
):
    with tempfile.TemporaryDirectory() as directory:
        data_dir = Path(directory)
        sessions = SessionManager(data_dir)
        vault = TokenVault(data_dir)
        model = ScriptedModel(
            [
                [function_call("auth_approval_demo", {}, call_id="call-both")],
                [assistant_message("done")],
            ]
        )
        factory = AgentFactory(service_config, model=model)
        formatter = CompleteFormatter(factory, sessions, vault)
        identity = make_context(
            subject_id="guardrail-user",
            roles={"auth_user"},
        ).identity
        try:
            first = await formatter.run(
                "Approve and authenticate",
                identity,
                request_id="req-1",
                session_id="session-2",
            )
            assert isinstance(first.outcome, TurnInterrupt)
            interrupt = first.outcome.interruptions[0]
            assert interrupt.requires_auth is True
            assert interrupt.requires_approval is True
            assert interrupt.authenticated is False
            assert interrupt.service == "auth_approval_demo"

            await vault.put(identity, "auth_approval_demo", "oauth-token")
            completed, _ = await formatter.resume(
                first.outcome.resume_token,
                {interrupt.interruption_id: "approve"},
                identity,
                request_id="req-2",
            )
            assert isinstance(completed.outcome, TurnComplete)
            assert completed.outcome.output == "done"
            model.assert_complete()
        finally:
            await factory.close()


@pytest.mark.asyncio
async def test_hitl_auth_required_marks_authenticated_when_vault_has_token(
    service_config,
):
    with tempfile.TemporaryDirectory() as directory:
        data_dir = Path(directory)
        sessions = SessionManager(data_dir)
        vault = TokenVault(data_dir)
        identity = make_context(
            subject_id="guardrail-user",
            roles={"auth_user"},
        ).identity
        await vault.put(identity, "auth_approval_demo", "oauth-token")
        model = ScriptedModel(
            [
                [function_call("auth_approval_demo", {}, call_id="call-both")],
                [assistant_message("done")],
            ]
        )
        factory = AgentFactory(service_config, model=model)
        formatter = CompleteFormatter(factory, sessions, vault)
        try:
            first = await formatter.run(
                "Approve and authenticate",
                identity,
                request_id="req-1",
                session_id="session-3",
            )
            assert isinstance(first.outcome, TurnInterrupt)
            interrupt = first.outcome.interruptions[0]
            assert interrupt.requires_auth is True
            assert interrupt.requires_approval is True
            assert interrupt.authenticated is True

            completed, _ = await formatter.resume(
                first.outcome.resume_token,
                {interrupt.interruption_id: "approve"},
                identity,
                request_id="req-2",
            )
            assert isinstance(completed.outcome, TurnComplete)
            assert completed.outcome.output == "done"
            model.assert_complete()
        finally:
            await factory.close()
