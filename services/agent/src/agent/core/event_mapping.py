"""Normalize OpenAI Agents SDK results into service-owned event models."""

import json
from collections.abc import Mapping
from typing import Any

from agents import (
    AgentUpdatedStreamEvent,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
    StreamEvent,
    ToolApprovalItem,
)

from agent.core.models import (
    AgentChanged,
    ApprovalRequest,
    ReasoningChunk,
    RunEvent,
    TextChunk,
    ToolResult,
    ToolStart,
    UsageUpdate,
)


def _value(raw: Any, name: str, default: Any = None) -> Any:
    if isinstance(raw, Mapping):
        return raw.get(name, default)
    return getattr(raw, name, default)


def _arguments(raw_arguments: Any) -> dict[str, Any]:
    if isinstance(raw_arguments, Mapping):
        return dict(raw_arguments)
    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return {"raw": raw_arguments}
        return dict(parsed) if isinstance(parsed, Mapping) else {"value": parsed}
    return {} if raw_arguments is None else {"value": raw_arguments}


def map_stream_event(
    event: StreamEvent,
    *,
    previous_agent_name: str | None = None,
) -> list[RunEvent]:
    """Map one SDK stream event to zero or more protocol-neutral events."""

    if isinstance(event, RawResponsesStreamEvent):
        event_type = _value(event.data, "type", "")
        delta = _value(event.data, "delta")
        if event_type == "response.output_text.delta" and isinstance(delta, str):
            return [TextChunk(delta=delta)]

        if event_type in {
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        } and isinstance(delta, str):
            return [ReasoningChunk(delta=delta)]

        return []

    if isinstance(event, AgentUpdatedStreamEvent):
        return [
            AgentChanged(
                agent_name=event.new_agent.name,
                previous_agent_name=previous_agent_name,
            )
        ]

    if not isinstance(event, RunItemStreamEvent):
        return []

    item = event.item
    raw_item = getattr(item, "raw_item", None)
    if event.name == "tool_called":
        return [
            ToolStart(
                call_id=str(_value(raw_item, "call_id", _value(raw_item, "id", ""))),
                tool_name=str(_value(raw_item, "name", "unknown_tool")),
                arguments=_arguments(_value(raw_item, "arguments")),
            )
        ]

    if event.name == "tool_output":
        return [
            ToolResult(
                call_id=str(_value(raw_item, "call_id", "")),
                tool_name=str(
                    _value(
                        raw_item,
                        "name",
                        getattr(
                            getattr(item, "tool_origin", None),
                            "tool_name",
                            "unknown_tool",
                        ),
                    )
                ),
                output=getattr(item, "output", _value(raw_item, "output")),
            )
        ]

    return []


def map_usage(usage: Any) -> UsageUpdate:
    return UsageUpdate(
        requests=int(getattr(usage, "requests", 0)),
        input_tokens=int(getattr(usage, "input_tokens", 0)),
        output_tokens=int(getattr(usage, "output_tokens", 0)),
        total_tokens=int(getattr(usage, "total_tokens", 0)),
    )


def approval_id(item: ToolApprovalItem, index: int) -> str:
    return item.call_id or f"approval-{index}"


def map_approvals(items: list[ToolApprovalItem]) -> tuple[ApprovalRequest, ...]:
    return tuple(
        ApprovalRequest(
            interruption_id=approval_id(item, index),
            tool_name=item.name or "unknown_tool",
            arguments=_arguments(item.arguments),
            agent_name=item.agent.name,
        )
        for index, item in enumerate(items)
    )
