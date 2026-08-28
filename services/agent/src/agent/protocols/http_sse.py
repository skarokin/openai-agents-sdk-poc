"""SSE protocol adapter for normalized root and nested agent events."""

from collections.abc import AsyncIterator
from typing import Annotated
from uuid import uuid4

from agent_common import (
    AgentResumeRequest,
    AgentRunRequest,
    EventSource,
    StreamEvent,
)
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from agent.core.models import IdentityContext
from agent.core.observability import bind_observability_context
from agent.core.session_manager import SessionManager
from agent.formatters import EventsFormatter

from .dependencies import events_formatter, session_manager, trusted_identity
from .mapping import stream_event

router = APIRouter(prefix="/v1/agent", tags=["agent-sse"])


def _encode_sse(event: StreamEvent) -> str:
    return (
        f"id: {event.sequence}\n"
        f"event: {event.event_type}\n"
        f"data: {event.model_dump_json()}\n\n"
    )


def _response(events: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/runs/stream")
async def stream_agent(
    body: AgentRunRequest,
    identity: Annotated[IdentityContext, Depends(trusted_identity)],
    formatter: Annotated[EventsFormatter, Depends(events_formatter)],
) -> StreamingResponse:
    request_id = uuid4().hex
    session_id = body.session_id or uuid4().hex

    async def events() -> AsyncIterator[str]:
        with bind_observability_context(identity, request_id):
            async for envelope in formatter.stream(
                body.input,
                identity,
                request_id=request_id,
                session_id=session_id,
            ):
                event = stream_event(
                    envelope,
                    request_id=request_id,
                    session_id=session_id,
                )
                yield _encode_sse(event)

    return _response(events())


@router.post("/runs/stream/resume")
async def resume_stream(
    body: AgentResumeRequest,
    identity: Annotated[IdentityContext, Depends(trusted_identity)],
    formatter: Annotated[EventsFormatter, Depends(events_formatter)],
    sessions: Annotated[SessionManager, Depends(session_manager)],
) -> StreamingResponse:
    request_id = uuid4().hex
    try:
        session_id = sessions.run_state_session_id(
            body.resume_token,
            identity.subject_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    decisions = {item.interruption_id: item.decision for item in body.decisions}
    if len(decisions) != len(body.decisions):
        raise HTTPException(status_code=400, detail="Duplicate interruption IDs")

    async def events() -> AsyncIterator[str]:
        try:
            with bind_observability_context(identity, request_id):
                async for envelope, resumed_session_id in formatter.resume(
                    body.resume_token,
                    decisions,
                    identity,
                    request_id=request_id,
                ):
                    event = stream_event(
                        envelope,
                        request_id=request_id,
                        session_id=resumed_session_id,
                    )
                    yield _encode_sse(event)
        except ValueError as exc:
            error = StreamEvent(
                sequence=1,
                request_id=request_id,
                session_id=session_id,
                source=EventSource(
                    agent_name="agent-service",
                    invocation_id=request_id,
                ),
                event_type="turn_error",
                data={
                    "code": type(exc).__name__,
                    "message": str(exc),
                    "retryable": False,
                },
            )
            yield _encode_sse(error)

    return _response(events())
