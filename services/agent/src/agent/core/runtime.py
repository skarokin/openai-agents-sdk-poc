"""Shared runtime construction for formatters."""

import time

from agents import RunConfig

from agent.core.auth import TokenVault
from agent.core.models import AgentContext, EventSink, IdentityContext
from agent.hooks import SOFT_DEADLINE_SECONDS


def create_agent_context(
    identity: IdentityContext,
    event_sink: EventSink,
    *,
    request_id: str,
    session_id: str,
    token_vault: TokenVault,
) -> AgentContext:
    return AgentContext(
        identity=identity,
        event_sink=event_sink,
        request_id=request_id,
        session_id=session_id,
        deadline_epoch_seconds=time.time() + SOFT_DEADLINE_SECONDS,
        token_vault=token_vault,
    )


def create_run_config(context: AgentContext) -> RunConfig:
    return RunConfig(
        workflow_name="multi-protocol-agent",
        group_id=context.session_id,
        trace_include_sensitive_data=False,
        trace_metadata={
            "request_id": context.request_id,
            "subject_id": context.identity.subject_id,
            "actor_id": context.identity.actor_id or "",
            "roles": ",".join(sorted(context.identity.roles)),
        },
    )
