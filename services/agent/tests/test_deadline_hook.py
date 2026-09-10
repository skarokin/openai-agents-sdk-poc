"""Soft/hard deadline controls."""

import threading
import time
from types import SimpleNamespace

import pytest
from conftest import make_context
from strands.hooks import BeforeModelCallEvent

from agent.controls.deadlines import (
    GLOBAL_DEADLINE_HOOK,
    SOFT_DEADLINE_MESSAGE,
    SoftDeadlineHook,
    start_hard_deadline_watchdog,
)
from agent.core.agent_factory import AgentFactory


def test_soft_deadline_sets_event_cancel():
    hook = SoftDeadlineHook()
    agent = SimpleNamespace(name="root", state={})
    event = BeforeModelCallEvent(
        agent=agent,  # type: ignore[arg-type]
        invocation_state={
            "soft_deadline_epoch_seconds": time.time() - 1,
            "hard_deadline_epoch_seconds": time.time() + 30,
            "request_id": "r1",
        },
    )
    hook.on_before_model(event)
    assert event.cancel == SOFT_DEADLINE_MESSAGE


def test_soft_deadline_noop_when_hard_already_due():
    hook = SoftDeadlineHook()
    agent = SimpleNamespace(name="root", state={})
    past = time.time() - 5
    event = BeforeModelCallEvent(
        agent=agent,  # type: ignore[arg-type]
        invocation_state={
            "soft_deadline_epoch_seconds": past,
            "hard_deadline_epoch_seconds": past,
        },
    )
    hook.on_before_model(event)
    assert event.cancel is False


def test_soft_deadline_noop_before_soft():
    hook = SoftDeadlineHook()
    agent = SimpleNamespace(name="root", state={})
    event = BeforeModelCallEvent(
        agent=agent,  # type: ignore[arg-type]
        invocation_state={
            "soft_deadline_epoch_seconds": time.time() + 60,
            "hard_deadline_epoch_seconds": time.time() + 90,
        },
    )
    hook.on_before_model(event)
    assert event.cancel is False


def test_hard_deadline_watchdog_sets_cancel_signal():
    signal = threading.Event()
    timer = start_hard_deadline_watchdog(signal, time.time() - 0.01)
    try:
        assert signal.wait(timeout=1.0)
    finally:
        timer.cancel()


@pytest.mark.asyncio
async def test_root_and_nested_attach_soft_deadline_hook(
    service_config, tmp_path, monkeypatch
):
    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path))
    factory = AgentFactory(service_config)
    try:
        context = make_context(roles={"calculator", "subagent_user"})
        root = await factory.create(context)
        nested = next(
            tool._nested_agent
            for tool in root.tool_registry.registry.values()
            if getattr(tool, "tool_name", None) == "open_subagent"
        )

        def has_deadline(agent) -> bool:
            callbacks = list(
                agent.hooks.get_callbacks_for(BeforeModelCallEvent(agent=agent))
            )
            return any(
                getattr(cb, "__self__", None) is GLOBAL_DEADLINE_HOOK for cb in callbacks
            )

        assert has_deadline(root)
        assert has_deadline(nested)
    finally:
        await factory.close()


@pytest.mark.asyncio
async def test_subagent_forwards_cancel_signal(service_config, tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path))
    from agent.core.agent_factory import as_subagent_tool, openai_model
    from agent.core.sessions import make_file_session_manager
    from agent.tools import builtin
    from strands import Agent

    captured: dict[str, object] = {}
    nested = Agent(
        name="Nested",
        system_prompt="test",
        description="test",
        tools=[builtin.calculator],
        model=openai_model("unused"),
        callback_handler=None,
    )
    tool = as_subagent_tool(
        nested,
        name="open_subagent",
        hitl_tools=service_config.hitl_tools,
        session_manager=make_file_session_manager("cancel-forward", "test-user"),
    )
    nested_runtime = tool._nested_agent
    signal = threading.Event()

    async def capture_stream(prompt, *, invocation_state=None, cancel_signal=None, **kwargs):
        captured["cancel_signal"] = cancel_signal
        captured["invocation_state"] = dict(invocation_state or {})
        yield {
            "result": SimpleNamespace(
                stop_reason="end_turn",
                interrupts=None,
                message={"role": "assistant", "content": [{"text": "ok"}]},
            )
        }

    nested_runtime.stream_async = capture_stream  # type: ignore[method-assign]
    parent = SimpleNamespace(
        _interrupt_state=SimpleNamespace(interrupts={}, activated=False),
        cancel_signal=threading.Event(),
    )

    async for _ in tool.stream(
        {
            "toolUseId": "t1",
            "name": "open_subagent",
            "input": {"input": "hi"},
        },
        {
            "token_vault": object(),
            "identity": object(),
            "agent": parent,
            "soft_deadline_epoch_seconds": time.time() + 60,
            "hard_deadline_epoch_seconds": time.time() + 90,
            "cancel_signal": signal,
        },
    ):
        pass

    assert captured["cancel_signal"] is signal
    assert captured["invocation_state"]["cancel_signal"] is signal
    assert "soft_deadline_epoch_seconds" in captured["invocation_state"]
    assert "hard_deadline_epoch_seconds" in captured["invocation_state"]
