"""Normalize Strands Agents results into service-owned event models."""

import json
from collections.abc import Mapping
from typing import Any

from strands.interrupt import Interrupt

from agent.controls.hitl import is_nested_hitl_reason
from agent.core.models import (
    AgentContext,
    ApprovalRequest,
    ReasoningChunk,
    RunEvent,
    TextChunk,
    ToolResult,
    ToolStart,
    UsageUpdate,
)


def _arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
        return dict(parsed) if isinstance(parsed, Mapping) else {"value": parsed}
    return {} if raw is None else {"value": raw}


def map_stream_event(
    event: Mapping[str, Any],
    *,
    tool_calls: dict[str, str] | None = None,
) -> list[RunEvent]:
    """Map one Strands stream event dict to zero or more protocol-neutral events."""

    if "data" in event and isinstance(event.get("data"), str):
        if event.get("reasoning"):
            return []
        return [TextChunk(delta=event["data"])]

    if event.get("reasoning") and isinstance(event.get("reasoningText"), str):
        return [ReasoningChunk(delta=event["reasoningText"])]

    tool_use = event.get("current_tool_use")
    if isinstance(tool_use, Mapping) and tool_use.get("name"):
        call_id = str(tool_use.get("toolUseId") or tool_use.get("tool_use_id") or "")
        tool_name = str(tool_use["name"])
        # current_tool_use is re-emitted on every input delta; only announce once.
        if not call_id:
            return []
        if tool_calls is not None:
            if call_id in tool_calls:
                return []
            tool_calls[call_id] = tool_name
        return [
            ToolStart(
                call_id=call_id,
                tool_name=tool_name,
                arguments=_arguments(tool_use.get("input")),
            )
        ]

    message = event.get("message")
    if isinstance(message, Mapping) and message.get("role") == "user":
        content = message.get("content")
        if isinstance(content, list):
            results: list[RunEvent] = []
            for block in content:
                if not isinstance(block, Mapping) or "toolResult" not in block:
                    continue
                tool_result = block["toolResult"]
                if not isinstance(tool_result, Mapping):
                    continue
                call_id = str(tool_result.get("toolUseId") or "")
                tool_name = (
                    tool_calls.get(call_id, "")
                    if tool_calls is not None
                    else ""
                )
                results.append(
                    ToolResult(
                        call_id=call_id,
                        tool_name=tool_name,
                        output=tool_result.get("content"),
                    )
                )
            return results

    return []


def map_usage(metrics: Any) -> UsageUpdate:
    usage = getattr(metrics, "accumulated_usage", None) or {}
    if isinstance(usage, Mapping):
        return UsageUpdate(
            requests=int(getattr(metrics, "cycle_count", 0) or 0),
            input_tokens=int(usage.get("inputTokens", 0) or 0),
            output_tokens=int(usage.get("outputTokens", 0) or 0),
            total_tokens=int(usage.get("totalTokens", 0) or 0),
        )
    return UsageUpdate(
        requests=int(getattr(metrics, "cycle_count", 0) or 0),
        input_tokens=int(getattr(usage, "inputTokens", 0) or 0),
        output_tokens=int(getattr(usage, "outputTokens", 0) or 0),
        total_tokens=int(getattr(usage, "totalTokens", 0) or 0),
    )


def tool_use_id_from_interrupt_id(interrupt_id: str) -> str | None:
    """Extract toolUseId from a BeforeToolCall interrupt id.

    Format: ``v1:before_tool_call:{toolUseId}:{uuid}``.
    """

    parts = interrupt_id.split(":")
    if len(parts) >= 4 and parts[0] == "v1" and parts[1] == "before_tool_call":
        return parts[2]
    return None


def tool_use_from_message(
    tool_use_message: Mapping[str, Any] | None,
    interrupt_id: str,
) -> tuple[str, dict[str, Any]]:
    """Resolve tool name/args from the interrupt-state assistant tool_use message."""

    tool_use_id = tool_use_id_from_interrupt_id(interrupt_id)
    if tool_use_message is None or not tool_use_id:
        return "", {}

    content = tool_use_message.get("content")
    if not isinstance(content, list):
        return "", {}

    for block in content:
        if not isinstance(block, Mapping) or "toolUse" not in block:
            continue
        tool_use = block["toolUse"]
        if not isinstance(tool_use, Mapping):
            continue
        if str(tool_use.get("toolUseId") or "") != tool_use_id:
            continue
        return str(tool_use.get("name") or ""), _arguments(tool_use.get("input"))

    return "", {}


def map_interrupts(
    interrupts: list[Interrupt] | tuple[Interrupt, ...] | None,
    *,
    agent_context: AgentContext | None = None,
    agent_name: str | None = None,
    tool_use_message: Mapping[str, Any] | None = None,
) -> tuple[ApprovalRequest, ...]:
    """Map Strands Interrupt objects into protocol ApprovalRequest models."""

    if not interrupts:
        return ()

    approvals: list[ApprovalRequest] = []
    for item in interrupts:
        reason = item.reason
        if isinstance(reason, Mapping) and reason.get("requires_auth"):
            service = str(reason.get("service") or "")
            challenge_url = reason.get("authorization_url")
            authenticated = False
            if agent_context is not None and service:
                authenticated = (
                    agent_context.token_vault.peek(agent_context.identity, service)
                    is not None
                )
            approvals.append(
                ApprovalRequest(
                    interruption_id=item.id,
                    tool_name=str(reason.get("tool_name") or item.name),
                    arguments=_arguments(reason.get("arguments")),
                    agent_name=str(reason.get("agent_name") or agent_name or "") or None,
                    requires_auth=True,
                    requires_approval=bool(reason.get("requires_approval", False)),
                    authenticated=authenticated,
                    service=service or None,
                    authorization_url=(
                        str(challenge_url) if challenge_url is not None else None
                    ),
                )
            )
            continue

        if is_nested_hitl_reason(reason):
            approvals.append(
                ApprovalRequest(
                    interruption_id=item.id,
                    tool_name=str(reason.get("tool_name") or ""),
                    arguments=_arguments(reason.get("arguments")),
                    agent_name=str(reason.get("agent_name") or agent_name or "") or None,
                    requires_auth=False,
                    requires_approval=True,
                )
            )
            continue

        tool_name, arguments = tool_use_from_message(tool_use_message, item.id)
        approvals.append(
            ApprovalRequest(
                interruption_id=item.id,
                tool_name=tool_name,
                arguments=arguments,
                agent_name=agent_name,
                requires_auth=False,
                requires_approval=True,
            )
        )

    return tuple(approvals)


def decision_to_interrupt_response(interrupt_id: str, decision: str) -> dict[str, Any]:
    """Convert HTTP approve/reject into Strands interruptResponse content."""

    if decision == "approve":
        response: Any = "yes"
    elif decision == "reject":
        response = "n"
    else:
        raise ValueError(f"Unsupported decision: {decision}")

    return {
        "interruptResponse": {
            "interruptId": interrupt_id,
            "response": response,
        }
    }


def final_output_text(message: Any) -> str:
    """Extract assistant text from an AgentResult message."""

    if message is None:
        return ""
    if isinstance(message, str):
        return message
    content = (
        message.get("content")
        if isinstance(message, Mapping)
        else getattr(message, "content", None)
    )
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(message)
    parts: list[str] = []
    for block in content:
        if isinstance(block, Mapping) and "text" in block:
            parts.append(str(block["text"]))
        elif hasattr(block, "get") and block.get("text"):
            parts.append(str(block.get("text")))
    return "".join(parts)
