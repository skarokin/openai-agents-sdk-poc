"""Stream event mapping tests."""

from agent.core.event_mapping import map_stream_event


def test_tool_start_uses_name_from_current_tool_use():
    tool_calls: dict[str, str] = {}
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
    tool_calls: dict[str, str] = {}
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
    tool_calls: dict[str, str] = {}
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


def test_current_tool_use_without_id_is_ignored():
    assert (
        map_stream_event(
            {"current_tool_use": {"name": "auth_approval_demo", "input": {}}},
            tool_calls={},
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
        tool_calls={},
    )
    assert results[0].call_id == "call-9"
    assert results[0].tool_name == ""


def test_text_chunk_mapping():
    events = map_stream_event({"data": "hello"})
    assert len(events) == 1
    assert events[0].delta == "hello"
