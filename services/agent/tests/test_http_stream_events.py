"""HTTP stream event mapping for annotation / guardrail."""

from agent.common.http_mapping import stream_event
from agent.core.models import (
    Annotation,
    EventEnvelope,
    EventSource,
    GuardrailTripped,
)


def _envelope(event) -> EventEnvelope:
    return EventEnvelope(
        sequence=1,
        source=EventSource(agent_name="root", invocation_id="req-1"),
        event=event,
    )


def test_stream_event_annotation():
    mapped = stream_event(
        _envelope(Annotation(kind="citation", data={"uri": "https://example.com"})),
        request_id="req-1",
        session_id="sess-1",
    )
    assert mapped.event_type == "annotation"
    assert mapped.data == {
        "kind": "citation",
        "data": {"uri": "https://example.com"},
    }


def test_stream_event_guardrail_tripped():
    mapped = stream_event(
        _envelope(GuardrailTripped(message="blocked", name="content_filter")),
        request_id="req-1",
        session_id="sess-1",
    )
    assert mapped.event_type == "guardrail_tripped"
    assert mapped.data["message"] == "blocked"
    assert mapped.data["name"] == "content_filter"
