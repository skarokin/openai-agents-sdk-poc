"""Deterministic verification that nested HITL bubbles to the root run."""

import tempfile
from pathlib import Path

import pytest
from agents import RunConfig, Runner
from agents.testing import ScriptedModel, assistant_message, function_call
from conftest import CollectingSink, make_context

from agent.core.agent_factory import AgentFactory
from agent.core.models import TextChunk
from agent.core.runtime import create_run_config
from agent.core.session_manager import SessionManager
from agent.hooks import GLOBAL_DEADLINE_HOOK, SOFT_DEADLINE_MESSAGE


@pytest.mark.asyncio
async def test_soft_deadline_steers_model_without_failing_run(service_config):
    model = ScriptedModel([[assistant_message("deadline fallback")]])
    factory = AgentFactory(service_config, model=model)
    sink = CollectingSink()
    context = make_context(
        subject_id="deadline-user",
        roles={"calculator"},
        event_sink=sink,
        request_id="deadline-request",
        session_id="deadline-session",
        deadline_offset_seconds=-1,
    )
    agent = await factory.create(context)
    run_config = create_run_config(context)
    run_config.tracing_disabled = True

    result = await Runner.run(
        agent,
        "Calculate after the deadline",
        context=context,
        hooks=GLOBAL_DEADLINE_HOOK,
        run_config=run_config,
    )

    assert result.final_output == "deadline fallback"
    assert model.first_call is not None
    assert any(
        isinstance(item, dict) and item.get("content") == SOFT_DEADLINE_MESSAGE
        for item in model.first_call.input
    )
    model.assert_complete()


@pytest.mark.asyncio
async def test_nested_approvals_bubble_to_root_state(service_config):
    model = ScriptedModel(
        [
            [
                function_call(
                    "protected_subagent",
                    {"input": "Call approval_demo"},
                    call_id="call-protected",
                )
            ],
            [function_call("approval_demo", {}, call_id="call-nested-approval")],
            [assistant_message("nested complete")],
            [assistant_message("root complete")],
        ]
    )
    factory = AgentFactory(service_config, model=model)
    sink = CollectingSink()
    context = make_context(
        roles={"calculator", "approval_user", "subagent_user"},
        event_sink=sink,
    )
    agent = await factory.create(context)
    run_config = RunConfig(tracing_disabled=True)

    first = await Runner.run(
        agent,
        "Use the protected subagent",
        context=context,
        hooks=GLOBAL_DEADLINE_HOOK,
        run_config=run_config,
    )
    assert [item.name for item in first.interruptions] == ["protected_subagent"]

    with tempfile.TemporaryDirectory() as directory:
        sessions = SessionManager(Path(directory))
        first_token = await sessions.save_run_state(
            first.to_state(),
            session_id=context.session_id,
            owner_subject_id=context.identity.subject_id,
        )
        with pytest.raises(PermissionError):
            sessions.run_state_session_id(first_token, "different-user")

        agent = await factory.create(context)
        first_state, _ = await sessions.load_run_state(
            first_token,
            agent=agent,
            context=context,
        )
        sessions.apply_decisions(first_state, {"call-protected": "approve"})
        second = await Runner.run(
            agent,
            first_state,
            context=context,
            hooks=GLOBAL_DEADLINE_HOOK,
            run_config=run_config,
        )
        assert [item.name for item in second.interruptions] == ["approval_demo"], (
            f"final={second.final_output!r}, items={second.new_items!r}, "
            f"remaining={model.remaining_steps}"
        )

        second_token = await sessions.save_run_state(
            second.to_state(),
            session_id=context.session_id,
            owner_subject_id=context.identity.subject_id,
        )
        agent = await factory.create(context)
        second_state, _ = await sessions.load_run_state(
            second_token,
            agent=agent,
            context=context,
        )
        sessions.apply_decisions(
            second_state,
            {"call-nested-approval": "approve"},
        )
        final = await Runner.run(
            agent,
            second_state,
            context=context,
            hooks=GLOBAL_DEADLINE_HOOK,
            run_config=run_config,
        )

    assert not final.interruptions
    assert final.final_output == "root complete"
    assert any(
        isinstance(event, TextChunk)
        and event.delta == "nested complete"
        and source.kind == "agent_tool"
        for event, source in sink.events
    )
    model.assert_complete()


@pytest.mark.asyncio
async def test_open_subagent_only_interrupts_for_nested_tool(service_config):
    model = ScriptedModel(
        [
            [
                function_call(
                    "open_subagent",
                    {"input": "Call approval_demo"},
                    call_id="call-open",
                )
            ],
            [function_call("approval_demo", {}, call_id="call-open-nested-approval")],
            [assistant_message("open nested complete")],
            [assistant_message("open root complete")],
        ]
    )
    factory = AgentFactory(service_config, model=model)
    context = make_context(
        roles={"calculator", "approval_user", "subagent_user"},
        request_id="open-test-request",
        session_id="open-test-session",
    )
    agent = await factory.create(context)
    run_config = RunConfig(tracing_disabled=True)

    first = await Runner.run(
        agent,
        "Use the open subagent",
        context=context,
        hooks=GLOBAL_DEADLINE_HOOK,
        run_config=run_config,
    )
    assert [item.name for item in first.interruptions] == ["approval_demo"]

    state = first.to_state()
    state.approve(first.interruptions[0])
    final = await Runner.run(
        agent,
        state,
        context=context,
        hooks=GLOBAL_DEADLINE_HOOK,
        run_config=run_config,
    )
    assert not final.interruptions
    assert final.final_output == "open root complete"
    model.assert_complete()


@pytest.mark.asyncio
async def test_local_mcp_requires_approval_only_for_delete_note(service_config):
    model = ScriptedModel(
        [
            [function_call("list_notes", {}, call_id="call-list")],
            [
                function_call(
                    "delete_note",
                    {"note_id": "1"},
                    call_id="call-delete",
                )
            ],
            [assistant_message("listed then deleted")],
        ]
    )
    factory = AgentFactory(service_config, model=model)
    try:
        denied = await factory.create(
            make_context(
                subject_id="mcp-user",
                roles={"calculator"},
                request_id="mcp-denied",
                session_id="mcp-denied",
            )
        )
        assert denied.mcp_servers == []
        assert factory._mcp_servers == {}

        context = make_context(
            subject_id="mcp-user",
            roles={"mcp_user"},
            request_id="mcp-request",
            session_id="mcp-session",
        )
        agent = await factory.create(context)
        run_config = RunConfig(tracing_disabled=True)

        first = await Runner.run(
            agent,
            "List notes, then delete note 1",
            context=context,
            run_config=run_config,
        )
        assert [item.name for item in first.interruptions] == ["delete_note"]

        state = first.to_state()
        state.approve(first.interruptions[0])
        final = await Runner.run(
            agent,
            state,
            context=context,
            run_config=run_config,
        )
        assert not final.interruptions
        assert final.final_output == "listed then deleted"
        model.assert_complete()
    finally:
        await factory.close()
