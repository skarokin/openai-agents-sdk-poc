"""Normalize Strands Agents results into service-owned event models."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from strands.interrupt import Interrupt

from agent.controls.hitl import is_nested_hitl_reason
from agent.core.models import (
    AgentContext,
    ApprovalRequest,
    ReasoningChunk,
    RunEvent,
    SubagentEvent,
    TextChunk,
    ToolResult,
    ToolStart,
    UsageUpdate,
)


SUBAGENT_EVENT_KEY = "subagent_event"


@dataclass
class _ToolCallTracker:
    """Per-stream: defer ToolStart until args are ready, emit once, join names on results."""

    names: dict[str, str] = field(default_factory=dict)
    started: set[str] = field(default_factory=set)

    def remember(self, call_id: str, tool_name: str) -> None:
        self.names[call_id] = tool_name

    def mark_started(self, call_id: str) -> None:
        self.started.add(call_id)

    def was_started(self, call_id: str) -> bool:
        return call_id in self.started

    def name_for(self, call_id: str) -> str:
        return self.names.get(call_id, "")


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


def _tool_args_ready(raw: Any) -> bool:
    """False while model is still streaming a partial tool-input string."""

    if isinstance(raw, Mapping):
        return True
    if isinstance(raw, str):
        if not raw.strip():
            return False
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return False
        return isinstance(parsed, Mapping)
    return raw is not None


def is_tool_activity_event(event: Mapping[str, Any]) -> bool:
    """True for tool start/result stream events"""

    tool_use = event.get("current_tool_use")
    if isinstance(tool_use, Mapping) and tool_use.get("name"):
        return True

    message = event.get("message")
    if isinstance(message, Mapping) and message.get("role") == "user":
        content = message.get("content")
        if isinstance(content, list):
            return any(
                isinstance(block, Mapping) and "toolResult" in block for block in content
            )
    return False


class StreamEventMapper:
    """Maps Strands stream events for one turn; owns tool-call dedupe state."""

    def __init__(self) -> None:
        self._tool_calls = _ToolCallTracker()

    def map(self, event: Mapping[str, Any]) -> list[RunEvent]:
        """Map one Strands stream event dict to zero or more protocol-neutral events."""

        stream = event.get("tool_stream_event")
        if isinstance(stream, Mapping):
            data = stream.get("data")
            if isinstance(data, Mapping) and data.get(SUBAGENT_EVENT_KEY):
                agent_name = str(data.get("agent_name") or "")
                inner = data.get("event")
                if not agent_name or not isinstance(inner, Mapping):
                    return []
                nested = self.map(inner)
                return [
                    SubagentEvent(agent_name=agent_name, event=item)
                    for item in nested
                    if isinstance(item, (ToolStart, ToolResult))
                ]
            return []

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
            raw_input = tool_use.get("input")
            # current_tool_use is re-emitted on every input delta; wait for parseable
            # args, then announce once with the full payload.
            if not call_id:
                return []
            self._tool_calls.remember(call_id, tool_name)
            if self._tool_calls.was_started(call_id):
                return []
            if not _tool_args_ready(raw_input):
                return []
            self._tool_calls.mark_started(call_id)
            return [
                ToolStart(
                    call_id=call_id,
                    tool_name=tool_name,
                    arguments=_arguments(raw_input),
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
                    results.append(
                        ToolResult(
                            call_id=call_id,
                            tool_name=self._tool_calls.name_for(call_id),
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


def tool_info_from_interrupt(interrupt: Interrupt) -> tuple[str, dict[str, Any]]:
    """Resolve tool name/args from the interrupt itself (no session context).

    Preference:
    1. Structured ``reason`` dict (auth / nested HITL) with ``tool_name`` / ``arguments``
    2. Native Strands HITL prompt string:
       ``Approve \"{tool_name}\"?\\n  Input: {json}``
    """

    reason = interrupt.reason
    if isinstance(reason, Mapping):
        tool_name = str(reason.get("tool_name") or "")
        if tool_name or "arguments" in reason:
            return tool_name, _arguments(reason.get("arguments"))

    if isinstance(reason, str) and reason.startswith('Approve "'):
        rest = reason[len('Approve "') :]
        tool_name, sep, after = rest.partition('"')
        if not sep:
            return "", {}
        marker = "\n  Input: "
        idx = after.find(marker)
        if idx < 0:
            return tool_name, {}
        raw = after[idx + len(marker) :]
        try:
            return tool_name, _arguments(json.loads(raw))
        except json.JSONDecodeError:
            return tool_name, {"raw": raw} if raw else {}

    return "", {}


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

        tool_name, arguments = tool_info_from_interrupt(item)
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
