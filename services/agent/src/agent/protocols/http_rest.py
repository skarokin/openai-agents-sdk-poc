"""REST protocol adapter for final agent outcomes."""

from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from agent.common.http_dependencies import complete_formatter, trusted_identity
from agent.common.http_mapping import complete_response
from agent.core.models import IdentityContext
from agent.core.observability import bind_observability_context
from agent.formatters import CompleteFormatter
from agent_common import AgentResponse, AgentResumeRequest, AgentRunRequest

router = APIRouter(prefix="/v1/agent", tags=["agent-rest"])


@router.post("/runs", response_model=AgentResponse)
async def run_agent(
    body: AgentRunRequest,
    identity: Annotated[IdentityContext, Depends(trusted_identity)],
    formatter: Annotated[CompleteFormatter, Depends(complete_formatter)],
):
    request_id = uuid4().hex
    session_id = body.session_id or uuid4().hex
    with bind_observability_context(identity, request_id):
        result = await formatter.run(
            body.input,
            identity,
            request_id=request_id,
            session_id=session_id,
        )

    return complete_response(
        result,
        request_id=request_id,
        session_id=session_id,
    )


@router.post("/runs/resume", response_model=AgentResponse)
async def resume_agent(
    body: AgentResumeRequest,
    identity: Annotated[IdentityContext, Depends(trusted_identity)],
    formatter: Annotated[CompleteFormatter, Depends(complete_formatter)],
):
    request_id = uuid4().hex
    decisions = {item.interruption_id: item.decision for item in body.decisions}

    if len(decisions) != len(body.decisions):
        raise HTTPException(status_code=400, detail="Duplicate interruption IDs")
    try:
        with bind_observability_context(identity, request_id):
            result, session_id = await formatter.resume(
                body.resume_token,
                decisions,
                identity,
                request_id=request_id,
            )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return complete_response(
        result,
        request_id=request_id,
        session_id=session_id,
    )
