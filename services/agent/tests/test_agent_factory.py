"""Identity-based tool graph construction."""

import time
import unittest

from agent.core.agent_factory import AgentFactory
from agent.core.config import load_service_config
from agent.core.models import AgentContext, IdentityContext
from agents import RunContextWrapper


class CollectingSink:
    async def emit(self, event, *, source) -> None:
        return None


class AgentFactoryTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.factory = AgentFactory(load_service_config())

    async def asyncTearDown(self):
        await self.factory.close()

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

    @staticmethod
    async def _enabled_names(agent, context: AgentContext) -> list[str]:
        tools = await agent.get_all_tools(RunContextWrapper(context))
        return [tool.name for tool in tools]

    async def test_unprivileged_caller_gets_no_tools(self):
        context = self._context()
        agent = await self.factory.create(context)
        self.assertEqual(
            {tool.name for tool in agent.tools},
            {
                "calculator",
                "approval_demo",
                "protected_subagent",
                "open_subagent",
            },
        )
        self.assertEqual(await self._enabled_names(agent, context), [])

    async def test_calculator_role_does_not_unlock_subagents(self):
        context = self._context("calculator")
        agent = await self.factory.create(context)
        self.assertEqual(
            await self._enabled_names(agent, context),
            ["calculator"],
        )

    async def test_full_roles_register_all_tools_from_global_config(self):
        context = self._context("calculator", "approval_user", "subagent_user")
        agent = await self.factory.create(context)
        self.assertEqual(
            set(await self._enabled_names(agent, context)),
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
        approval = next(tool for tool in agent.tools if tool.name == "approval_demo")
        self.assertTrue(approval.needs_approval)


if __name__ == "__main__":
    unittest.main()
