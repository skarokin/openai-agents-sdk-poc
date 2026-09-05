"""Run an agent to a final output or durable interruption."""

import logging

from strands import Agent

from agent.core.agent_factory import AgentFactory
from agent.core.event_mapping import (
    decision_to_interrupt_response,
    final_output_text,
    map_interrupts,
    map_usage,
)
from agent.core.models import (
    CompleteResult,
    IdentityContext,
    Query,
    TurnComplete,
    TurnError,
    TurnInterrupt,
)
from agent.core.observability import bind_observability_context
from agent.core.runtime import create_agent_context, create_limits
from agent.core.token_vault import TokenVault

logger = logging.getLogger(__name__)


class CompleteFormatter:
    def __init__(self, factory: AgentFactory, token_vault: TokenVault):
        self._factory = factory
        self._token_vault = token_vault

    async def _execute(
        self,
        query: Query,
        context,
        agent: Agent,
    ) -> CompleteResult:
        with bind_observability_context(context.identity, context.request_id):
            try:
                result = await agent.invoke_async(
                    query.input,
                    invocation_state=context.to_invocation_state(),
                    limits=create_limits(self._factory.max_turns),
                )
                if result.stop_reason == "interrupt":
                    return CompleteResult(
                        outcome=TurnInterrupt(
                            interruptions=map_interrupts(
                                result.interrupts,
                                agent_context=context,
                                agent_name=agent.name,
                                tool_use_message=agent._interrupt_state.context.get(
                                    "tool_use_message"
                                ),
                            ),
                        )
                    )

                return CompleteResult(
                    outcome=TurnComplete(
                        output=final_output_text(result.message),
                        usage=map_usage(result.metrics),
                    )
                )
            except Exception:
                logger.exception("agent run failed")
                return CompleteResult(
                    outcome=TurnError(
                        code="AgentExecutionError",
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
        context = create_agent_context(
            identity,
            request_id=request_id,
            session_id=session_id,
            token_vault=self._token_vault,
        )
        with bind_observability_context(identity, request_id):
            agent = await self._factory.create(context)
        return await self._execute(Query(input=input_text), context, agent)

    async def resume(
        self,
        session_id: str,
        decisions: dict[str, str],
        identity: IdentityContext,
        *,
        request_id: str,
    ) -> CompleteResult:
        context = create_agent_context(
            identity,
            request_id=request_id,
            session_id=session_id,
            token_vault=self._token_vault,
        )
        responses = [
            decision_to_interrupt_response(interrupt_id, decision)
            for interrupt_id, decision in decisions.items()
        ]
        with bind_observability_context(identity, request_id):
            agent = await self._factory.create(context)

        return await self._execute(Query(input=responses), context, agent)
