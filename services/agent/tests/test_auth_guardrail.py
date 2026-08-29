"""Guardrail-based auth checks."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from agents.testing import ScriptedModel, assistant_message, function_call
from agents.tool_guardrails import ToolInputGuardrailResult
from conftest import CollectingSink, make_context

from agent.core.agent_factory import AgentFactory
from agent.core.auth import (
    MODEL_AUTH_MESSAGE,
    AuthTokenMissingError,
    TokenVault,
    auth_challenges_from_guardrail_results,
    auth_guardrail,
    get_access_token,
)
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
async def test_model_sees_generic_message_not_url():
    sink = CollectingSink()
    context = make_context(
        subject_id="user",
        roles={"auth_user"},
        event_sink=sink,
        request_id="req",
        session_id="session",
        deadline_offset_seconds=9_999_999_999.0,
        token_vault=TokenVault(Path(tempfile.mkdtemp())),
    )
    guardrail = auth_guardrail("authentication_demo")
    data = MagicMock()
    data.context.context = context
    data.context.tool_name = "authentication_demo"
    data.context.tool_call_id = "call-1"
    data.agent = MagicMock(name="demo-agent")

    output = await guardrail.run(data)

    assert output.behavior["message"] == MODEL_AUTH_MESSAGE
    assert "agent://" not in output.behavior["message"]
    assert (
        output.output_info["authorization_url"] == "agent://vault/authentication_demo"
    )
    challenges = auth_challenges_from_guardrail_results(
        [ToolInputGuardrailResult(guardrail=guardrail, output=output)]
    )
    assert len(challenges) == 1
    assert challenges[0].authorization_url == "agent://vault/authentication_demo"


@pytest.mark.asyncio
async def test_auth_only_completes_with_auth_required(service_config):
    with tempfile.TemporaryDirectory() as directory:
        data_dir = Path(directory)
        sessions = SessionManager(data_dir)
        vault = TokenVault(data_dir)
        model = ScriptedModel(
            [
                [function_call("authentication_demo", {}, call_id="call-auth")],
                [assistant_message("auth blocked")],
            ]
        )
        factory = AgentFactory(service_config, model=model)
        formatter = CompleteFormatter(factory, sessions, vault)
        identity = make_context(
            subject_id="guardrail-user",
            roles={"auth_user"},
        ).identity
        try:
            result = await formatter.run(
                "Authenticate",
                identity,
                request_id="req-1",
                session_id="session-1",
            )
            assert isinstance(result.outcome, TurnComplete)
            assert len(result.outcome.auth_required) == 1
            challenge = result.outcome.auth_required[0]
            assert challenge.service == "authentication_demo"
            assert challenge.authorization_url == "agent://vault/authentication_demo"
            model.assert_complete()
        finally:
            await factory.close()


@pytest.mark.asyncio
async def test_auth_only_runs_when_vault_has_token(service_config):
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
            assert result.outcome.auth_required == ()
            assert result.outcome.output == "authenticated"
            model.assert_complete()
        finally:
            await factory.close()


@pytest.mark.asyncio
async def test_hitl_runs_before_auth_guardrail(service_config):
    with tempfile.TemporaryDirectory() as directory:
        data_dir = Path(directory)
        sessions = SessionManager(data_dir)
        vault = TokenVault(data_dir)
        model = ScriptedModel(
            [
                [function_call("auth_approval_demo", {}, call_id="call-both")],
                [assistant_message("auth blocked")],
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
            assert [item.tool_name for item in first.outcome.interruptions] == [
                "auth_approval_demo"
            ]

            approved, _ = await formatter.resume(
                first.outcome.resume_token,
                {first.outcome.interruptions[0].interruption_id: "approve"},
                identity,
                request_id="req-2",
            )
            assert isinstance(approved.outcome, TurnComplete)
            assert len(approved.outcome.auth_required) == 1
            assert approved.outcome.auth_required[0].service == "auth_approval_demo"
            model.assert_complete()
        finally:
            await factory.close()
