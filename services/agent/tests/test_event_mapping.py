"""Stream event mapping tests."""

from types import SimpleNamespace

from agents import RunItemStreamEvent

from agent.core.event_mapping import map_stream_event


def test_tool_output_uses_name_from_prior_tool_call():
    tool_calls: dict[str, str] = {}
    called = RunItemStreamEvent(
        name="tool_called",
        item=SimpleNamespace(
            raw_item={"call_id": "call-1", "name": "auth_approval_demo", "arguments": "{}"},
            tool_origin=None,
        ),
    )
    output = RunItemStreamEvent(
        name="tool_output",
        item=SimpleNamespace(
            raw_item={"call_id": "call-1", "output": "ok"},
            output="ok",
            tool_origin=None,
        ),
    )

    map_stream_event(called, tool_calls=tool_calls)
    results = map_stream_event(output, tool_calls=tool_calls)

    assert len(results) == 1
    assert results[0].tool_name == "auth_approval_demo"
    assert results[0].output == "ok"
