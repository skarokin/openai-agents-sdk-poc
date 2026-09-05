"""Shared runtime construction for formatters."""

import time
from typing import Any

from strands.types.agent import Limits

from agent.controls.hooks import SOFT_DEADLINE_SECONDS
from agent.core.models import AgentContext, IdentityContext
from agent.core.token_vault import TokenVault


def create_agent_context(
    identity: IdentityContext,
    *,
    request_id: str,
    session_id: str,
    token_vault: TokenVault,
) -> AgentContext:
    return AgentContext(
        identity=identity,
        request_id=request_id,
        session_id=session_id,
        deadline_epoch_seconds=time.time() + SOFT_DEADLINE_SECONDS,
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
