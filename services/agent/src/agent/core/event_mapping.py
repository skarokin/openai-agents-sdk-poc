"""Normalize Strands Agents results into service-owned event models."""

import json
import re
from collections.abc import Mapping
from typing import Any

from strands.interrupt import Interrupt

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

_APPROVE_PROMPT_RE = re.compile(r'^Approve "([^"]+)"')


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


def _tool_name_from_hitl_reason(reason: Any) -> str | None:
    if not isinstance(reason, str):
        return None
    match = _APPROVE_PROMPT_RE.match(reason.strip())
    return match.group(1) if match else None


def _arguments_from_hitl_reason(reason: Any) -> dict[str, Any]:
    if not isinstance(reason, str):
        return {}
    marker = "Input: "
    index = reason.rfind(marker)
    if index < 0:
        return {}
    return _arguments(reason[index + len(marker) :].strip())


def map_interrupts(
    interrupts: list[Interrupt] | tuple[Interrupt, ...] | None,
    *,
    agent_context: AgentContext | None = None,
    agent_name: str | None = None,
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
                    agent_name=agent_name,
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

        tool_name = _tool_name_from_hitl_reason(reason) or item.name
        approvals.append(
            ApprovalRequest(
                interruption_id=item.id,
                tool_name=tool_name,
                arguments=_arguments_from_hitl_reason(reason),
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
