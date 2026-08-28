"""Identity-based tool graph construction."""

import time
import unittest

from agent.core.agent_factory import AgentFactory
from agent.core.config import load_service_config
from agent.core.models import AgentContext, IdentityContext


class CollectingSink:
    async def emit(self, event, *, source) -> None:
        return None


class AgentFactoryTest(unittest.TestCase):
    def setUp(self):
        self.factory = AgentFactory(load_service_config())

    def _context(self, *roles: str) -> AgentContext:
        return AgentContext(
            identity=IdentityContext(
                subject_id="rbac-user",
                roles=frozenset(roles),
            ),
            event_sink=CollectingSink(),
            request_id="rbac-request",
            session_id="rbac-session",
            deadline_epoch_seconds=time.time() + 60,
        )

    def test_unprivileged_caller_gets_no_tools(self):
        agent = self.factory.create(self._context())
        self.assertEqual([tool.name for tool in agent.tools], [])

    def test_calculator_role_does_not_unlock_subagents(self):
        agent = self.factory.create(self._context("calculator"))
        self.assertEqual([tool.name for tool in agent.tools], ["calculator"])

    def test_full_roles_register_all_tools_from_global_config(self):
        agent = self.factory.create(
            self._context("calculator", "approval_user", "subagent_user")
        )
        self.assertEqual(
            {tool.name for tool in agent.tools},
            {
                "calculator",
                "approval_demo",
                "protected_subagent",
                "open_subagent",
            },
        )
        nested = next(tool for tool in agent.tools if tool.name == "open_subagent")
        self.assertFalse(nested.needs_approval)
        protected = next(
            tool for tool in agent.tools if tool.name == "protected_subagent"
        )
        self.assertTrue(protected.needs_approval)


if __name__ == "__main__":
    unittest.main()
