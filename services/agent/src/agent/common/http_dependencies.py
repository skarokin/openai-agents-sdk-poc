"""FastAPI dependencies shared by HTTP endpoints."""

from typing import Annotated

from fastapi import Header, HTTPException, Request

from agent.core.models import IdentityContext
from agent.runners import AgentRunner
from agent_common import ACTOR_HEADER, ROLES_HEADER, SUBJECT_HEADER


def trusted_identity(
    subject_id: Annotated[str | None, Header(alias=SUBJECT_HEADER)] = None,
    actor_id: Annotated[str | None, Header(alias=ACTOR_HEADER)] = None,
    roles: Annotated[str | None, Header(alias=ROLES_HEADER)] = None,
) -> IdentityContext:
    """Read identity asserted by the trusted edge authorizer."""

    if not subject_id:
        raise HTTPException(
            status_code=401,
            detail=f"Missing trusted identity header {SUBJECT_HEADER}",
        )
    parsed_roles = frozenset(
        role.strip() for role in (roles or "").split(",") if role.strip()
    )
    return IdentityContext(
        subject_id=subject_id,
        actor_id=actor_id,
        roles=parsed_roles,
    )


def agent_runner(request: Request) -> AgentRunner:
    return request.app.state.agent_runner
