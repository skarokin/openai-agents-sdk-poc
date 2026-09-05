"""Yield normalized stream events from a Strands agent run."""

import logging
from collections.abc import AsyncIterator

from strands import Agent

from agent.controls.deadlines import start_hard_deadline_watchdog
from agent.core.agent_factory import AgentFactory
from agent.core.event_mapping import (
    decision_to_interrupt_response,
    final_output_text,
    map_interrupts,
    map_stream_event,
    map_usage,
)
from agent.core.models import (
    EventEnvelope,
    EventSource,
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


class EventsFormatter:
    def __init__(self, factory: AgentFactory, token_vault: TokenVault):
        self._factory = factory
        self._token_vault = token_vault
        # Survives interrupt/resume streams so tool_result keeps the name from tool_start.
        self._tool_names: dict[str, dict[str, str]] = {}

    def _tool_names_for(self, context) -> dict[str, str]:
        key = f"{context.identity.subject_id}:{context.session_id}"
        return self._tool_names.setdefault(key, {})

    async def _stream_prepared(
        self,
        query: Query,
        context,
        agent: Agent,
    ) -> AsyncIterator[EventEnvelope]:
        sequence = 0
        source = EventSource(
            agent_name=agent.name or self._factory._config.agent.name,
            invocation_id=context.request_id,
            kind="root",
        )

        tool_calls = self._tool_names_for(context)
        result = None

        try:
            with bind_observability_context(context.identity, context.request_id):
                watchdog = start_hard_deadline_watchdog(
                    context.cancel_signal,
                    context.hard_deadline_epoch_seconds,
                )
                try:
                    async for event in agent.stream_async(
                        query.input,
                        invocation_state=context.to_invocation_state(),
                        limits=create_limits(self._factory.max_turns),
                        cancel_signal=context.cancel_signal,
                    ):
                        if "result" in event:
                            result = event["result"]
                            continue

                        for normalized in map_stream_event(event, tool_calls=tool_calls):
                            sequence += 1
                            yield EventEnvelope(
                                sequence=sequence,
                                source=source,
                                event=normalized,
                            )
                finally:
                    watchdog.cancel()

                if result is None:
                    raise RuntimeError("Strands stream ended without a result event")

                if result.stop_reason == "interrupt":
                    terminal = TurnInterrupt(
                        interruptions=map_interrupts(
                            result.interrupts,
                            agent_context=context,
                            agent_name=agent.name,
                            tool_use_message=agent._interrupt_state.context.get(
                                "tool_use_message"
                            ),
                        ),
                    )
                else:
                    key = f"{context.identity.subject_id}:{context.session_id}"
                    self._tool_names.pop(key, None)
                    terminal = TurnComplete(
                        output=final_output_text(result.message),
                        usage=map_usage(result.metrics),
                    )

                sequence += 1
                yield EventEnvelope(sequence=sequence, source=source, event=terminal)

        except Exception:
            logger.exception(
                "streaming agent run failed",
                extra={
                    "request_id": context.request_id,
                    "subject_id": context.identity.subject_id,
                    "actor_id": context.identity.actor_id or "-",
                },
            )
            sequence += 1
            yield EventEnvelope(
                sequence=sequence,
                source=source,
                event=TurnError(
                    code="AgentExecutionError",
                    message="Agent execution failed",
                    retryable=False,
                ),
            )

    async def stream(
        self,
        input_text: str,
        identity: IdentityContext,
        *,
        request_id: str,
        session_id: str,
    ) -> AsyncIterator[EventEnvelope]:
        context = create_agent_context(
            identity,
            request_id=request_id,
            session_id=session_id,
            token_vault=self._token_vault,
        )
        with bind_observability_context(identity, request_id):
            agent = await self._factory.create(context)

        async for event in self._stream_prepared(
            Query(input=input_text),
            context,
            agent,
        ):
            yield event

    async def resume(
        self,
        session_id: str,
        decisions: dict[str, str],
        identity: IdentityContext,
        *,
        request_id: str,
    ) -> AsyncIterator[EventEnvelope]:
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

        async for event in self._stream_prepared(
            Query(input=responses),
            context,
            agent,
        ):
            yield event
