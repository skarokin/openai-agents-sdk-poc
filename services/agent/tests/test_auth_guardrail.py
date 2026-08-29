"""Guardrail-based auth checks."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from agents.tool_guardrails import ToolInputGuardrailResult

from agent.core.agent_factory import AgentFactory
from agent.core.auth_guardrail import (
    MODEL_AUTH_MESSAGE,
    AuthTokenMissingError,
    auth_challenges_from_guardrail_results,
    auth_guardrail,
    get_access_token,
)
from agent.core.config import load_service_config
from agent.core.models import AgentContext, IdentityContext, TurnComplete, TurnInterrupt
from agent.core.session_manager import SessionManager
from agent.core.token_vault import TokenVault
from agent.formatters.complete import CompleteFormatter

try:
    from test_nested_hitl import SequenceModel, _message, _tool_call
except ImportError:
    from tests.test_nested_hitl import SequenceModel, _message, _tool_call


class CollectingSink:
    def __init__(self):
        self.events = []

    async def emit(self, event, *, source) -> None:
        self.events.append((event, source))


class AuthGuardrailUnitTest(unittest.IsolatedAsyncioTestCase):
    async def test_get_access_token_raises_when_vault_empty(self):
        vault = TokenVault(Path(tempfile.mkdtemp()))
        context = AgentContext(
            identity=IdentityContext(subject_id="user", roles=frozenset({"auth_user"})),
            event_sink=CollectingSink(),
            request_id="req",
            session_id="session",
            deadline_epoch_seconds=9_999_999_999.0,
            token_vault=vault,
        )

        with self.assertRaises(AuthTokenMissingError) as raised:
            await get_access_token(context, "authentication_demo")

        self.assertEqual(raised.exception.service, "authentication_demo")

    async def test_get_access_token_returns_token_when_present(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = TokenVault(Path(directory))
            identity = IdentityContext(
                subject_id="user", roles=frozenset({"auth_user"})
            )
            await vault.put(identity, "authentication_demo", "oauth-token")
            context = AgentContext(
                identity=identity,
                event_sink=CollectingSink(),
                request_id="req",
                session_id="session",
                deadline_epoch_seconds=9_999_999_999.0,
                token_vault=vault,
            )

            token = await get_access_token(context, "authentication_demo")

            self.assertEqual(token, "oauth-token")

    async def test_model_sees_generic_message_not_url(self):
        sink = CollectingSink()
        context = AgentContext(
            identity=IdentityContext(subject_id="user", roles=frozenset({"auth_user"})),
            event_sink=sink,
            request_id="req",
            session_id="session",
            deadline_epoch_seconds=9_999_999_999.0,
            token_vault=TokenVault(Path(tempfile.mkdtemp())),
        )
        guardrail = auth_guardrail("authentication_demo")
        data = MagicMock()
        data.context.context = context
        data.context.tool_name = "authentication_demo"
        data.context.tool_call_id = "call-1"
        data.agent = MagicMock(name="demo-agent")

        output = await guardrail.run(data)

        self.assertEqual(output.behavior["message"], MODEL_AUTH_MESSAGE)
        self.assertNotIn("agent://", output.behavior["message"])
        self.assertEqual(
            output.output_info["authorization_url"],
            "agent://vault/authentication_demo",
        )
        challenges = auth_challenges_from_guardrail_results(
            [ToolInputGuardrailResult(guardrail=guardrail, output=output)]
        )
        self.assertEqual(len(challenges), 1)
        self.assertEqual(
            challenges[0].authorization_url, "agent://vault/authentication_demo"
        )


class AuthGuardrailFormatterTest(unittest.IsolatedAsyncioTestCase):
    async def test_auth_only_completes_with_auth_required(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            sessions = SessionManager(data_dir)
            vault = TokenVault(data_dir)
            model = SequenceModel(
                [
                    [_tool_call("authentication_demo", "call-auth", {})],
                    [_message("auth blocked", "blocked-message")],
                ]
            )
            factory = AgentFactory(load_service_config(), model=model)
            formatter = CompleteFormatter(factory, sessions, vault)
            identity = IdentityContext(
                subject_id="guardrail-user",
                roles=frozenset({"auth_user"}),
            )
            try:
                result = await formatter.run(
                    "Authenticate",
                    identity,
                    request_id="req-1",
                    session_id="session-1",
                )
                self.assertIsInstance(result.outcome, TurnComplete)
                self.assertEqual(len(result.outcome.auth_required), 1)
                challenge = result.outcome.auth_required[0]
                self.assertEqual(challenge.service, "authentication_demo")
                self.assertEqual(
                    challenge.authorization_url,
                    "agent://vault/authentication_demo",
                )
            finally:
                await factory.close()

    async def test_auth_only_runs_when_vault_has_token(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            sessions = SessionManager(data_dir)
            vault = TokenVault(data_dir)
            identity = IdentityContext(
                subject_id="guardrail-user",
                roles=frozenset({"auth_user"}),
            )
            await vault.put(identity, "authentication_demo", "oauth-token")
            model = SequenceModel(
                [
                    [_tool_call("authentication_demo", "call-auth", {})],
                    [_message("authenticated", "auth-message")],
                ]
            )
            factory = AgentFactory(load_service_config(), model=model)
            formatter = CompleteFormatter(factory, sessions, vault)
            try:
                result = await formatter.run(
                    "Authenticate",
                    identity,
                    request_id="req-1",
                    session_id="session-1",
                )
                self.assertIsInstance(result.outcome, TurnComplete)
                self.assertEqual(result.outcome.auth_required, ())
                self.assertEqual(result.outcome.output, "authenticated")
            finally:
                await factory.close()

    async def test_hitl_runs_before_auth_guardrail(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            sessions = SessionManager(data_dir)
            vault = TokenVault(data_dir)
            model = SequenceModel(
                [
                    [_tool_call("auth_approval_demo", "call-both", {})],
                    [_message("auth blocked", "blocked-message")],
                ]
            )
            factory = AgentFactory(load_service_config(), model=model)
            formatter = CompleteFormatter(factory, sessions, vault)
            identity = IdentityContext(
                subject_id="guardrail-user",
                roles=frozenset({"auth_user"}),
            )
            try:
                first = await formatter.run(
                    "Approve and authenticate",
                    identity,
                    request_id="req-1",
                    session_id="session-2",
                )
                self.assertIsInstance(first.outcome, TurnInterrupt)
                self.assertEqual(
                    [item.tool_name for item in first.outcome.interruptions],
                    ["auth_approval_demo"],
                )

                approved, _ = await formatter.resume(
                    first.outcome.resume_token,
                    {first.outcome.interruptions[0].interruption_id: "approve"},
                    identity,
                    request_id="req-2",
                )
                self.assertIsInstance(approved.outcome, TurnComplete)
                self.assertEqual(len(approved.outcome.auth_required), 1)
                self.assertEqual(
                    approved.outcome.auth_required[0].service,
                    "auth_approval_demo",
                )
            finally:
                await factory.close()


if __name__ == "__main__":
    unittest.main()
