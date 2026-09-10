"""Stream event mapping tests."""

from types import SimpleNamespace

from strands.interrupt import Interrupt

from agent.core.event_mapping import (
    ToolCallTracker,
    decision_to_interrupt_response,
    final_output_text,
    map_interrupts,
    map_stream_event,
    map_usage,
    tool_use_from_message,
    tool_use_id_from_interrupt_id,
)
from agent.core.models import ReasoningChunk, SubagentEvent, TextChunk, ToolStart, UsageUpdate


def test_tool_start_uses_name_from_current_tool_use():
    tool_calls = ToolCallTracker()
    events = map_stream_event(
        {
            "current_tool_use": {
                "toolUseId": "call-1",
                "name": "auth_approval_demo",
                "input": {},
            }
        },
        tool_calls=tool_calls,
    )
    assert len(events) == 1
    assert events[0].tool_name == "auth_approval_demo"
    assert events[0].call_id == "call-1"


def test_tool_result_joins_name_from_earlier_tool_start_in_same_stream():
    tool_calls = ToolCallTracker()
    map_stream_event(
        {
            "current_tool_use": {
                "toolUseId": "call-1",
                "name": "auth_approval_demo",
                "input": {},
            }
        },
        tool_calls=tool_calls,
    )
    results = map_stream_event(
        {
            "message": {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "call-1",
                            "content": [{"text": "ok"}],
                        }
                    }
                ],
            }
        },
        tool_calls=tool_calls,
    )

    assert len(results) == 1
    assert results[0].tool_name == "auth_approval_demo"
    assert results[0].output == [{"text": "ok"}]


def test_current_tool_use_emits_tool_start_once():
    tool_calls = ToolCallTracker()
    first = {
        "current_tool_use": {
            "toolUseId": "call-1",
            "name": "auth_approval_demo",
            "input": {"partial": True},
        }
    }
    second = {
        "current_tool_use": {
            "toolUseId": "call-1",
            "name": "auth_approval_demo",
            "input": {"partial": True, "done": True},
        }
    }

    assert len(map_stream_event(first, tool_calls=tool_calls)) == 1
    assert map_stream_event(second, tool_calls=tool_calls) == []


def test_current_tool_use_waits_for_parseable_streaming_input():
    tool_calls = ToolCallTracker()
    empty = {
        "current_tool_use": {
            "toolUseId": "call-1",
            "name": "calculator",
            "input": "",
        }
    }
    partial = {
        "current_tool_use": {
            "toolUseId": "call-1",
            "name": "calculator",
            "input": '{"expression"',
        }
    }
    complete = {
        "current_tool_use": {
            "toolUseId": "call-1",
            "name": "calculator",
            "input": '{"expression": "5+5"}',
        }
    }

    assert map_stream_event(empty, tool_calls=tool_calls) == []
    assert map_stream_event(partial, tool_calls=tool_calls) == []
    events = map_stream_event(complete, tool_calls=tool_calls)
    assert len(events) == 1
    assert events[0].arguments == {"expression": "5+5"}
    assert map_stream_event(complete, tool_calls=tool_calls) == []


def test_current_tool_use_without_id_is_ignored():
    assert (
        map_stream_event(
            {"current_tool_use": {"name": "auth_approval_demo", "input": {}}},
            tool_calls=ToolCallTracker(),
        )
        == []
    )


def test_tool_result_without_prior_start_has_empty_name():
    results = map_stream_event(
        {
            "message": {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "call-9",
                            "content": [{"text": "done"}],
                        }
                    }
                ],
            }
        },
        tool_calls=ToolCallTracker(),
    )
    assert results[0].call_id == "call-9"
    assert results[0].tool_name == ""


def test_text_chunk_mapping():
    events = map_stream_event({"data": "hello"})
    assert len(events) == 1
    assert isinstance(events[0], TextChunk)
    assert events[0].delta == "hello"


def test_reasoning_text_is_ignored_when_flagged_as_data():
    assert map_stream_event({"data": "think", "reasoning": True}) == []


def test_reasoning_chunk_mapping():
    events = map_stream_event({"reasoning": True, "reasoningText": "step 1"})
    assert len(events) == 1
    assert isinstance(events[0], ReasoningChunk)
    assert events[0].delta == "step 1"


def test_unknown_stream_event_maps_to_empty():
    assert map_stream_event({"lifecycle": "start"}) == []


def test_subagent_tool_stream_maps_nested_tool_start_only():
    tool_calls = ToolCallTracker()
    events = map_stream_event(
        {
            "type": "tool_stream",
            "tool_stream_event": {
                "tool_use": {"toolUseId": "parent-1", "name": "open_subagent"},
                "data": {
                    "subagent_event": True,
                    "agent_name": "Open Subagent",
                    "event": {
                        "current_tool_use": {
                            "toolUseId": "nested-1",
                            "name": "calculator",
                            "input": {"expression": "5+5"},
                        }
                    },
                },
            },
        },
        tool_calls=tool_calls,
    )
    assert len(events) == 1
    assert isinstance(events[0], SubagentEvent)
    assert events[0].agent_name == "Open Subagent"
    assert isinstance(events[0].event, ToolStart)
    assert events[0].event.tool_name == "calculator"
    assert events[0].event.arguments == {"expression": "5+5"}


def test_subagent_tool_stream_ignores_nested_text():
    events = map_stream_event(
        {
            "tool_stream_event": {
                "tool_use": {"toolUseId": "parent-1", "name": "open_subagent"},
                "data": {
                    "subagent_event": True,
                    "agent_name": "Open Subagent",
                    "event": {"data": "thinking out loud"},
                },
            }
        }
    )
    # TextChunk would map from inner event, but SubagentEvent only keeps tool activity.
    assert events == []


def test_map_usage_from_accumulated_dict():
    metrics = SimpleNamespace(
        cycle_count=2,
        accumulated_usage={
            "inputTokens": 10,
            "outputTokens": 5,
            "totalTokens": 15,
        },
    )
    assert map_usage(metrics) == UsageUpdate(
        requests=2,
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
    )


def test_map_usage_defaults_when_missing():
    metrics = SimpleNamespace(cycle_count=0, accumulated_usage=None)
    assert map_usage(metrics) == UsageUpdate()


def test_final_output_text_from_message_blocks():
    message = {
        "role": "assistant",
        "content": [{"text": "Hello"}, {"text": " world"}],
    }
    assert final_output_text(message) == "Hello world"


def test_final_output_text_from_string_and_empty():
    assert final_output_text("plain") == "plain"
    assert final_output_text(None) == ""


def test_decision_to_interrupt_response():
    assert decision_to_interrupt_response("i1", "approve") == {
        "interruptResponse": {"interruptId": "i1", "response": "yes"}
    }
    assert decision_to_interrupt_response("i1", "reject") == {
        "interruptResponse": {"interruptId": "i1", "response": "n"}
    }


def test_tool_use_id_from_interrupt_id():
    assert (
        tool_use_id_from_interrupt_id("v1:before_tool_call:tu-1:deadbeef") == "tu-1"
    )
    assert tool_use_id_from_interrupt_id("not-a-hitl-id") is None


def test_tool_use_from_message_matches_interrupt_id():
    message = {
        "role": "assistant",
        "content": [
            {
                "toolUse": {
                    "toolUseId": "tu-1",
                    "name": "approval_demo",
                    "input": {"x": 1},
                }
            }
        ],
    }
    name, args = tool_use_from_message(
        message, "v1:before_tool_call:tu-1:aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    )
    assert name == "approval_demo"
    assert args == {"x": 1}


def test_native_hitl_uses_tool_use_message_not_reason_string():
    interrupt = Interrupt(
        id="v1:before_tool_call:tu-9:deadbeef",
        name="strands:human-in-the-loop",
        reason='Approve "ignored"?\n  Input: {"nope": true}',
    )
    mapped = map_interrupts(
        [interrupt],
        agent_name="root",
        tool_use_message={
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "tu-9",
                        "name": "approval_demo",
                        "input": {"ok": True},
                    }
                }
            ],
        },
    )
    assert mapped[0].tool_name == "approval_demo"
    assert mapped[0].arguments == {"ok": True}
