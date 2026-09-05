"""Identity-based tool graph construction."""

import pytest
from conftest import make_context

from agent.core.agent_factory import AgentFactory


@pytest.fixture
async def factory(service_config):
    instance = AgentFactory(service_config)
    yield instance
    await instance.close()


@pytest.mark.asyncio
async def test_unprivileged_caller_gets_no_tools(factory):
    context = make_context()
    agent = await factory.create(context)
    assert agent.tool_names == []


@pytest.mark.asyncio
async def test_calculator_role_does_not_unlock_subagents(factory):
    context = make_context(roles={"calculator"})
    agent = await factory.create(context)
    assert agent.tool_names == ["calculator"]


@pytest.mark.asyncio
async def test_full_roles_register_catalog_tools(factory, service_config):
    context = make_context(
        roles={"calculator", "approval_user", "subagent_user", "auth_user"},
    )
    agent = await factory.create(context)
    assert set(agent.tool_names) == {
        "calculator",
        "approval_demo",
        "protected_subagent",
        "open_subagent",
        "authentication_demo",
        "auth_approval_demo",
    }
    assert "approval_demo" in service_config.hitl_tools
    assert "protected_subagent" in service_config.hitl_tools
    assert "open_subagent" not in service_config.hitl_tools
    assert "delete_note" in service_config.mcp_hitl_tools


@pytest.mark.asyncio
async def test_create_builds_fresh_agent_each_call(factory):
    """Session persistence is on disk; create() must not cache Agent instances."""

    context = make_context(
        roles={"calculator", "approval_user", "subagent_user"},
        session_id="session-no-cache",
    )
    first = await factory.create(context)
    second = await factory.create(context)
    assert first is not second
    assert first.agent_id == second.agent_id == "root"
