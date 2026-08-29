"""Map internal dataclasses to shared HTTP Pydantic contracts."""

from dataclasses import asdict
from typing import Any

from agent.core.models import (
    AgentChanged,
    AuthRequired,
    CompleteResult,
    EventEnvelope,
    GuardrailTripped,
    ReasoningChunk,
    TextChunk,
    ToolResult,
    ToolStart,
    TurnComplete,
    TurnError,
    TurnInterrupt,
    UsageUpdate,
)
from agent_common import (
    ApprovalRequest as HttpApprovalRequest,
)
from agent_common import (
    AuthRequiredInfo,
    CompletedResponse,
    ErrorResponse,
    InterruptedResponse,
    StreamEvent,
    Usage,
)
from agent_common import (
    EventSource as HttpEventSource,
)


def _usage(usage: UsageUpdate) -> Usage:
    return Usage(
        requests=usage.requests,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
    )


def _approvals(
    interruptions,
) -> tuple[HttpApprovalRequest, ...]:
    return tuple(
        HttpApprovalRequest(
            interruption_id=item.interruption_id,
            tool_name=item.tool_name,
            arguments=dict(item.arguments),
            agent_name=item.agent_name,
            requires_auth=item.requires_auth,
            requires_approval=item.requires_approval,
            authenticated=item.authenticated,
            service=item.service,
            authorization_url=item.authorization_url,
        )
        for item in interruptions
    )


def _auth_required(
    challenges,
) -> tuple[AuthRequiredInfo, ...]:
    return tuple(
        AuthRequiredInfo(
            call_id=item.call_id,
            tool_name=item.tool_name,
            service=item.service,
            authorization_url=item.authorization_url,
        )
        for item in challenges
    )


def complete_response(
    result: CompleteResult,
    *,
    request_id: str,
    session_id: str,
):
    outcome = result.outcome
    if isinstance(outcome, TurnComplete):
        return CompletedResponse(
            request_id=request_id,
            session_id=session_id,
            output=outcome.output,
            usage=_usage(outcome.usage),
            auth_required=_auth_required(outcome.auth_required),
        )

    if isinstance(outcome, TurnInterrupt):
        if outcome.resume_token is None:
            raise RuntimeError("Interrupted result was not persisted")

        return InterruptedResponse(
            request_id=request_id,
            session_id=session_id,
            resume_token=outcome.resume_token,
            interruptions=_approvals(outcome.interruptions),
        )

    return ErrorResponse(
        request_id=request_id,
        session_id=session_id,
        code=outcome.code,
        message=outcome.message,
        retryable=outcome.retryable,
    )


def stream_event(
    envelope: EventEnvelope,
    *,
    request_id: str,
    session_id: str,
) -> StreamEvent:
    event = envelope.event
    event_type: str
    data: dict[str, Any]
    if isinstance(event, TextChunk):
        event_type, data = "text_chunk", {"delta": event.delta}
    elif isinstance(event, ReasoningChunk):
        event_type, data = "reasoning_chunk", {"delta": event.delta}
    elif isinstance(event, ToolStart):
        event_type, data = (
            "tool_start",
            {
                "call_id": event.call_id,
                "tool_name": event.tool_name,
                "arguments": dict(event.arguments),
            },
        )
    elif isinstance(event, ToolResult):
        event_type, data = (
            "tool_result",
            {
                "call_id": event.call_id,
                "tool_name": event.tool_name,
                "output": event.output,
            },
        )
    elif isinstance(event, GuardrailTripped):
        event_type, data = "guardrail_tripped", asdict(event)
    elif isinstance(event, AuthRequired):
        event_type, data = "auth_required", asdict(event)
    elif isinstance(event, UsageUpdate):
        event_type, data = "usage_update", asdict(event)
    elif isinstance(event, AgentChanged):
        event_type, data = "agent_changed", asdict(event)
    elif isinstance(event, TurnComplete):
        event_type, data = (
            "turn_complete",
            {
                "output": event.output,
                "usage": asdict(event.usage),
                "auth_required": [asdict(item) for item in event.auth_required],
            },
        )
    elif isinstance(event, TurnInterrupt):
        if event.resume_token is None:
            raise RuntimeError("Interrupted event was not persisted")
        event_type, data = (
            "turn_interrupt",
            {
                "resume_token": str(event.resume_token),
                "interruptions": [
                    {
                        "interruption_id": item.interruption_id,
                        "tool_name": item.tool_name,
                        "arguments": dict(item.arguments),
                        "agent_name": item.agent_name,
                        "requires_auth": item.requires_auth,
                        "requires_approval": item.requires_approval,
                        "authenticated": item.authenticated,
                        "service": item.service,
                        "authorization_url": item.authorization_url,
                    }
                    for item in event.interruptions
                ],
            },
        )
    elif isinstance(event, TurnError):
        event_type, data = "turn_error", asdict(event)
    else:
        raise TypeError(f"Unsupported event: {type(event).__name__}")

    return StreamEvent(
        sequence=envelope.sequence,
        request_id=request_id,
        session_id=session_id,
        source=HttpEventSource(
            agent_name=envelope.source.agent_name,
            invocation_id=envelope.source.invocation_id,
            parent_invocation_id=envelope.source.parent_invocation_id,
            kind=envelope.source.kind,
        ),
        event_type=event_type,  # type: ignore[arg-type]
        data=data,
    )
