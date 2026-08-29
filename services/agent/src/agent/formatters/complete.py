"""Run an agent to a final output or durable interruption."""

import logging
from uuid import UUID

from agents import Agent, Runner

from agent.core.agent_factory import AgentFactory
from agent.controls.guardrails import TokenVault
from agent.core.event_mapping import map_approvals, map_usage
from agent.core.models import (
    AgentContext,
    CompleteResult,
    IdentityContext,
    NoOpEventSink,
    Query,
    TurnComplete,
    TurnError,
    TurnInterrupt,
)
from agent.core.observability import bind_observability_context
from agent.core.runtime import create_agent_context, create_run_config
from agent.core.session_manager import SessionManager
from agent.controls.hooks import GLOBAL_DEADLINE_HOOK

logger = logging.getLogger(__name__)


class CompleteFormatter:
    def __init__(
        self,
        factory: AgentFactory,
        sessions: SessionManager,
        token_vault: TokenVault,
    ):
        self._factory = factory
        self._sessions = sessions
        self._token_vault = token_vault

    async def _execute(
        self,
        query: Query,
        context: AgentContext,
        agent: Agent[AgentContext],
    ) -> CompleteResult:
        with bind_observability_context(
            context.identity,
            context.request_id,
        ):
            try:
                result = await Runner.run(
                    agent,
                    query.input,
                    context=context,
                    hooks=GLOBAL_DEADLINE_HOOK,
                    run_config=create_run_config(context),
                    max_turns=self._factory.max_turns,
                    session=self._sessions.session(
                        context.session_id,
                        context.identity.subject_id,
                    ),
                )
                if result.interruptions:
                    state = result.to_state()
                    token = await self._sessions.save_run_state(
                        state,
                        session_id=context.session_id,
                        owner_subject_id=context.identity.subject_id,
                    )
                    return CompleteResult(
                        outcome=TurnInterrupt(
                            interruptions=map_approvals(
                                result.interruptions,
                                agent_context=context,
                                run_context=result.context_wrapper,
                            ),
                            run_state=state,
                            resume_token=token,
                        )
                    )

                return CompleteResult(
                    outcome=TurnComplete(
                        output=result.final_output,
                        usage=map_usage(result.context_wrapper.usage),
                    )
                )
            except Exception as exc:
                logger.exception("agent run failed")
                return CompleteResult(
                    outcome=TurnError(
                        code=type(exc).__name__,
                        message="Agent execution failed",
                        retryable=False,
                    )
                )

    async def run(
        self,
        input_text: str,
        identity: IdentityContext,
        *,
        request_id: str,
        session_id: str,
    ) -> CompleteResult:
        """Start a new agent run"""

        context = create_agent_context(
            identity,
            NoOpEventSink(),
            request_id=request_id,
            session_id=session_id,
            token_vault=self._token_vault,
        )

        with bind_observability_context(identity, request_id):
            agent = await self._factory.create(context)

        return await self._execute(Query(input=input_text), context, agent)

    async def resume(
        self,
        token: UUID,
        decisions: dict[str, str],
        identity: IdentityContext,
        *,
        request_id: str,
    ) -> tuple[CompleteResult, str]:
        """Resume an interrupted agent run"""

        async with self._sessions.lock_run_state(token):
            session_id = self._sessions.run_state_session_id(
                token,
                identity.subject_id,
            )
            context = create_agent_context(
                identity,
                NoOpEventSink(),
                request_id=request_id,
                session_id=session_id,
                token_vault=self._token_vault,
            )
            with bind_observability_context(identity, request_id):
                agent = await self._factory.create(context)
            state, _ = await self._sessions.load_run_state(
                token,
                agent=agent,
                context=context,
            )
            if state.get_interruptions():
                self._sessions.apply_decisions(state, decisions)
            result = await self._execute(Query(input=state), context, agent)
            if not isinstance(result.outcome, TurnError):
                await self._sessions.delete_run_state(token)
            return result, session_id
