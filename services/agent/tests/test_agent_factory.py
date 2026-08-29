"""Identity-based tool graph construction."""

import pytest
from agents import RunContextWrapper
from conftest import make_context

from agent.core.agent_factory import AgentFactory
from agent.core.models import AgentContext


@pytest.fixture
async def factory(service_config):
    instance = AgentFactory(service_config)
    yield instance
    await instance.close()


async def _enabled_names(agent, context: AgentContext) -> list[str]:
    tools = await agent.get_all_tools(RunContextWrapper(context))
    return [tool.name for tool in tools]


@pytest.mark.asyncio
async def test_unprivileged_caller_gets_no_tools(factory):
    context = make_context()
    agent = await factory.create(context)
    assert {tool.name for tool in agent.tools} == {
        "calculator",
        "approval_demo",
        "protected_subagent",
        "open_subagent",
        "authentication_demo",
        "auth_approval_demo",
    }
    assert await _enabled_names(agent, context) == []


@pytest.mark.asyncio
async def test_calculator_role_does_not_unlock_subagents(factory):
    context = make_context(roles={"calculator"})
    agent = await factory.create(context)
    assert await _enabled_names(agent, context) == ["calculator"]


@pytest.mark.asyncio
async def test_full_roles_register_all_tools_from_global_config(factory):
    context = make_context(
        roles={"calculator", "approval_user", "subagent_user", "auth_user"},
    )
    agent = await factory.create(context)
    assert set(await _enabled_names(agent, context)) == {
        "calculator",
        "approval_demo",
        "protected_subagent",
        "open_subagent",
        "authentication_demo",
        "auth_approval_demo",
    }
    nested = next(tool for tool in agent.tools if tool.name == "open_subagent")
    assert not nested.needs_approval
    protected = next(tool for tool in agent.tools if tool.name == "protected_subagent")
    assert protected.needs_approval
    approval = next(tool for tool in agent.tools if tool.name == "approval_demo")
    assert approval.needs_approval
