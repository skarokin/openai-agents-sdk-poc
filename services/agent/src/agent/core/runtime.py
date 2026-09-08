"""Shared request/context construction for runners."""

import time
from typing import Any

from strands.types.agent import Limits

from agent.controls.deadlines import HARD_DEADLINE_SECONDS, SOFT_DEADLINE_SECONDS
from agent.core.models import AgentContext, IdentityContext
from agent.core.token_vault import TokenVault


def create_agent_context(
    identity: IdentityContext,
    *,
    request_id: str,
    session_id: str,
    token_vault: TokenVault,
) -> AgentContext:
    now = time.time()
    return AgentContext(
        identity=identity,
        request_id=request_id,
        session_id=session_id,
        soft_deadline_epoch_seconds=now + SOFT_DEADLINE_SECONDS,
        hard_deadline_epoch_seconds=now + HARD_DEADLINE_SECONDS,
        token_vault=token_vault,
    )


def create_limits(max_turns: int) -> Limits:
    return Limits(turns=max_turns)


def create_trace_attributes(context: AgentContext) -> dict[str, Any]:
    """Trace metadata attached at Agent construction."""

    return {
        "workflow.name": "multi-protocol-agent",
        "session.id": context.session_id,
        "request_id": context.request_id,
        "subject_id": context.identity.subject_id,
        "actor_id": context.identity.actor_id or "",
        "roles": ",".join(sorted(context.identity.roles)),
    }
