"""Normalize OpenAI Agents SDK results into service-owned event models."""

import json
from collections.abc import Mapping
from typing import Any

from agents import (
    AgentUpdatedStreamEvent,
    RawResponsesStreamEvent,
    RunContextWrapper,
    RunItemStreamEvent,
    StreamEvent,
    ToolApprovalItem,
)

from agent.core.auth.auth_approval import approval_policy_for_item
from agent.core.models import (
    AgentChanged,
    AgentContext,
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


def _resolve_tool_name(
    item: Any,
    raw_item: Any,
    *,
    tool_calls: Mapping[str, str] | None = None,
) -> str:
    name = _value(raw_item, "name", None)
    if isinstance(name, str) and name:
        return name

    call_id = str(_value(raw_item, "call_id", _value(raw_item, "id", "")))
    if tool_calls and call_id in tool_calls:
        return tool_calls[call_id]

    tool_origin = getattr(item, "tool_origin", None)
    if tool_origin is not None:
        origin_name = getattr(tool_origin, "agent_tool_name", None)
        if isinstance(origin_name, str) and origin_name:
            return origin_name

    return "unknown_tool"


def map_stream_event(
    event: StreamEvent,
    *,
    previous_agent_name: str | None = None,
    tool_calls: dict[str, str] | None = None,
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
        call_id = str(_value(raw_item, "call_id", _value(raw_item, "id", "")))
        tool_name = _resolve_tool_name(item, raw_item, tool_calls=tool_calls)
        if tool_calls is not None and call_id:
            tool_calls[call_id] = tool_name

        return [
            ToolStart(
                call_id=call_id,
                tool_name=tool_name,
                arguments=_arguments(_value(raw_item, "arguments")),
            )
        ]

    if event.name == "tool_output":
        return [
            ToolResult(
                call_id=str(_value(raw_item, "call_id", _value(raw_item, "id", ""))),
                tool_name=_resolve_tool_name(item, raw_item, tool_calls=tool_calls),
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


def map_approvals(
    items: list[ToolApprovalItem],
    *,
    agent_context: AgentContext | None = None,
    run_context: RunContextWrapper[AgentContext] | None = None,
) -> tuple[ApprovalRequest, ...]:
    approvals: list[ApprovalRequest] = []

    for index, item in enumerate(items):
        tool_name = item.name or "unknown_tool"
        call_id = approval_id(item, index)
        policy = approval_policy_for_item(item)
        service = policy.vault_service if policy.requires_auth else None
        authenticated = False
        challenge = None
        if policy.requires_auth and agent_context is not None and service is not None:
            authenticated = (
                agent_context.token_vault.peek(agent_context.identity, service)
                is not None
            )
            challenge = agent_context.token_vault.challenge(service)

        approvals.append(
            ApprovalRequest(
                interruption_id=call_id,
                tool_name=tool_name,
                arguments=_arguments(item.arguments),
                agent_name=item.agent.name,
                requires_auth=policy.requires_auth,
                requires_approval=policy.requires_approval,
                authenticated=authenticated,
                service=challenge.service if challenge is not None else None,
                authorization_url=(
                    challenge.authorization_url if challenge is not None else None
                ),
            )
        )
    return tuple(approvals)
