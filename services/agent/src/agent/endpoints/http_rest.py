"""REST endpoint for final agent outcomes."""

from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from agent.common.http_dependencies import agent_runner, trusted_identity
from agent.common.http_mapping import complete_response
from agent.core.models import IdentityContext
from agent.runners import AgentRunner
from agent_common import AgentResponse, AgentResumeRequest, AgentRunRequest

router = APIRouter(prefix="/v1/agent", tags=["agent-rest"])


@router.post("/runs", response_model=AgentResponse)
async def run_agent(
    body: AgentRunRequest,
    identity: Annotated[IdentityContext, Depends(trusted_identity)],
    runner: Annotated[AgentRunner, Depends(agent_runner)],
):
    request_id = uuid4().hex
    session_id = body.session_id or uuid4().hex
    result = await runner.sync_run(
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
    runner: Annotated[AgentRunner, Depends(agent_runner)],
):
    request_id = uuid4().hex
    decisions = {item.interruption_id: item.decision for item in body.decisions}

    if len(decisions) != len(body.decisions):
        raise HTTPException(status_code=400, detail="Duplicate interruption IDs")
    try:
        result = await runner.sync_resume(
            body.session_id,
            decisions,
            identity,
            request_id=request_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return complete_response(
        result,
        request_id=request_id,
        session_id=body.session_id,
    )
