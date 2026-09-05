"""Config-driven HITL catalogs and nested interrupt coverage."""

from types import SimpleNamespace

import pytest
from conftest import make_context
from strands.interrupt import Interrupt, InterruptException
from strands.vended_interventions.hitl import HumanInTheLoop

from agent.config import hitl_allowed_tools, load_service_config
from agent.controls.interrupts import NESTED_HITL_REASON_KEY, require_vault_token
from agent.core.agent_factory import AgentFactory, _subagent_tool
from agent.core.event_mapping import map_interrupts
from agent.core.models import SubagentSpec
from agent.core.token_vault import TokenVault
from agent.tools import builtin


def _hitl_handlers(agent) -> list[HumanInTheLoop]:
    registry = agent._intervention_registry
    return [item for item in registry.handlers if isinstance(item, HumanInTheLoop)]


def test_config_builds_hitl_and_mcp_hitl_catalogs(service_config):
    assert service_config.hitl_tools == frozenset(
        {
            "approval_demo",
            "auth_approval_demo",
            "protected_subagent",
        }
    )
    assert service_config.mcp_hitl_tools == frozenset({"delete_note"})
    assert service_config.all_hitl_tools == (
        service_config.hitl_tools | service_config.mcp_hitl_tools
    )
    assert hitl_allowed_tools(service_config.all_hitl_tools) == [
        "*",
        "!approval_demo",
        "!auth_approval_demo",
        "!delete_note",
        "!protected_subagent",
    ]


@pytest.mark.asyncio
async def test_parent_hitl_includes_mcp_and_catalog_tools(service_config):
    factory = AgentFactory(service_config)
    try:
        # Skip mcp_user so this test doesn't need a live notes MCP process.
        context = make_context(
            roles={
                "calculator",
                "approval_user",
                "subagent_user",
                "auth_user",
            },
        )
        agent = await factory.create(context)
        handlers = _hitl_handlers(agent)
        assert len(handlers) == 1
        allowed = set(handlers[0]._allowed_tools)
        assert "*" in allowed
        assert "!approval_demo" in allowed
        assert "!auth_approval_demo" in allowed
        assert "!protected_subagent" in allowed
        # MCP hitl tools are still wired into the parent intervention from config.
        assert "!delete_note" in allowed
        assert "!open_subagent" not in allowed
        assert "!authentication_demo" not in allowed
    finally:
        await factory.close()


def test_subagent_hitl_covers_nested_leaf_tools(service_config, tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path))
    spec = SubagentSpec(
        agent_name="Nested",
        instructions="test",
        description="test",
        tools=("calculator", "approval_demo", "authentication_demo"),
    )
    from agent.core.sessions import make_file_session_manager

    nested_tool = _subagent_tool(
        spec,
        name="open_subagent",
        tools=[builtin.calculator, builtin.approval_demo, builtin.authentication_demo],
        model="unused",
        hitl_tools=service_config.hitl_tools,
        session_manager=make_file_session_manager("hitl-spec", "test-user"),
    )
    handlers = _hitl_handlers(nested_tool._nested_agent)
    assert len(handlers) == 1
    allowed = set(handlers[0]._allowed_tools)
    # Nested HITL only for leaf tools marked hitl:true in config.
    assert "!approval_demo" in allowed
    assert "!auth_approval_demo" not in allowed
    assert "!authentication_demo" not in allowed
    assert "!protected_subagent" not in allowed


@pytest.mark.asyncio
async def test_nested_interrupt_restored_via_shared_session(
    service_config, tmp_path, monkeypatch
):
    """Fresh create() restores nested interrupt state from the shared session."""

    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path))
    factory = AgentFactory(service_config)
    try:
        context = make_context(
            roles={"calculator", "approval_user", "subagent_user"},
            session_id="nested-session-restore",
        )
        parent = await factory.create(context)
        nested_tool = next(
            tool
            for tool in parent.tool_registry.registry.values()
            if getattr(tool, "tool_name", None) == "open_subagent"
        )
        nested_agent = nested_tool._nested_agent
        assert nested_agent.agent_id == "open_subagent"
        assert nested_agent._session_manager is parent._session_manager

        interrupt = Interrupt(
            id="nested-hitl-1",
            name="strands:human-in-the-loop",
            reason='Approve "approval_demo"?\n  Input: {}',
        )
        nested_agent._interrupt_state.interrupts[interrupt.id] = interrupt
        nested_agent._interrupt_state.activate()
        nested_agent._session_manager.sync_agent(nested_agent)

        resumed_parent = await factory.create(context)
        assert resumed_parent is not parent
        resumed_nested = next(
            tool
            for tool in resumed_parent.tool_registry.registry.values()
            if getattr(tool, "tool_name", None) == "open_subagent"
        )._nested_agent
        assert resumed_nested is not nested_agent
        assert resumed_nested._interrupt_state.activated
        assert "nested-hitl-1" in resumed_nested._interrupt_state.interrupts
        assert (
            resumed_nested._interrupt_state.interrupts["nested-hitl-1"].reason
            == interrupt.reason
        )
    finally:
        await factory.close()


@pytest.mark.asyncio
async def test_subagent_tool_forwards_invocation_state(service_config, tmp_path, monkeypatch):
    """Custom subagent tool passes ToolContext.invocation_state into invoke_async."""

    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path))
    from agent.core.sessions import make_file_session_manager

    captured: dict[str, object] = {}
    spec = SubagentSpec(
        agent_name="Nested",
        instructions="test",
        description="test",
        tools=("calculator",),
    )
    tool = _subagent_tool(
        spec,
        name="open_subagent",
        tools=[builtin.calculator],
        model="unused",
        hitl_tools=service_config.hitl_tools,
        session_manager=make_file_session_manager("forward-state", "test-user"),
    )
    nested = tool._nested_agent

    async def capture_invoke(prompt, *, invocation_state=None, **kwargs):
        captured["prompt"] = prompt
        captured["invocation_state"] = dict(invocation_state or {})
        return SimpleNamespace(
            stop_reason="end_turn",
            interrupts=None,
            message={"role": "assistant", "content": [{"text": "ok"}]},
        )

    nested.invoke_async = capture_invoke  # type: ignore[method-assign]
    vault = object()
    identity = object()
    parent = SimpleNamespace(
        _interrupt_state=SimpleNamespace(interrupts={}, activated=False)
    )

    results = []
    async for event in tool.stream(
        {
            "toolUseId": "t1",
            "name": "open_subagent",
            "input": {"input": "hi"},
        },
        {"token_vault": vault, "identity": identity, "agent": parent},
    ):
        results.append(event)

    assert captured["prompt"] == "hi"
    assert captured["invocation_state"]["token_vault"] is vault
    assert captured["invocation_state"]["identity"] is identity
    assert "agent" not in captured["invocation_state"]
    assert results, "expected a tool result event"


@pytest.mark.asyncio
async def test_auth_interrupt_from_within_nested_agent_context(tmp_path):
    """Auth interrupt is raised by the tool body, independent of HITL."""

    vault = TokenVault(tmp_path)
    identity = make_context(roles={"auth_user"}, token_vault=vault).identity
    nested_agent = SimpleNamespace(
        name="Open Subagent",
        state={"identity": {"subject_id": identity.subject_id, "roles": ["auth_user"]}},
    )

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
        tool_use_message={
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "call-hitl",
                        "name": "auth_approval_demo",
                        "input": {},
                    }
                }
            ],
        },
    )
    assert mapped[0].requires_approval is True and mapped[0].requires_auth is False
    assert mapped[0].tool_name == "auth_approval_demo"
    assert mapped[1].requires_auth is True and mapped[1].requires_approval is False


def test_nested_hitl_interrupt_carries_subagent_name():
    bubbled = Interrupt(
        id="n1",
        name="nested-id",
        reason={
            NESTED_HITL_REASON_KEY: True,
            "agent_name": "Protected Subagent",
            "tool_name": "approval_demo",
            "arguments": {},
        },
    )
    mapped = map_interrupts([bubbled], agent_name="Multi-Protocol Assistant")
    assert mapped[0].agent_name == "Protected Subagent"
    assert mapped[0].tool_name == "approval_demo"
    assert mapped[0].requires_approval is True
    assert mapped[0].requires_auth is False


def test_mcp_hitl_tool_is_config_driven_not_hardcoded():
    config = load_service_config()
    assert "delete_note" in config.mcp_hitl_tools
    assert "list_notes" not in config.mcp_hitl_tools
