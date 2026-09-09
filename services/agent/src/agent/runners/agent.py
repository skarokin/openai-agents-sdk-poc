"""
Runs the agent for one turn.

Endpoints call this instead of talking to Strands directly.

- stream_run / stream_resume: yield EventEnvelope as the turn progresses
- sync_run / sync_resume: consume that same stream and return the terminal CompleteResult
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

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
    AgentContext,
    CompleteResult,
    EventEnvelope,
    EventSource,
    IdentityContext,
    TurnComplete,
    TurnError,
    TurnInterrupt,
    TurnOutcome,
)
from agent.core.observability import bind_observability_context
from agent.core.runtime import create_agent_context, create_limits
from agent.core.token_vault import TokenVault

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Query:
    """Input for either a new agent run or a resumed interrupted run."""

    input: str | list[dict[str, Any]]

    @property
    def is_resume(self) -> bool:
        return isinstance(self.input, list)


class AgentRunner:
    """Single streaming turn path; sync APIs collect the terminal event."""

    def __init__(self, factory: AgentFactory, token_vault: TokenVault) -> None:
        self._factory = factory
        self._token_vault = token_vault
        # Survives interrupt/resume streams so tool_result keeps the name from tool_start.
        self._tool_names: dict[str, dict[str, str]] = {}

    def _tool_names_for(self, context: AgentContext) -> dict[str, str]:
        key = f"{context.identity.subject_id}:{context.session_id}"
        return self._tool_names.setdefault(key, {})

    async def _prepare(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
        session_id: str,
    ) -> tuple[AgentContext, Agent]:
        """
        create the agent with required context, bindings, etc
        """
        context = create_agent_context(
            identity,
            request_id=request_id,
            session_id=session_id,
            token_vault=self._token_vault,
        )
        agent = await self._factory.create(context)
        return context, agent

    async def _stream_turn(
        self,
        query: _Query,
        identity: IdentityContext,
        *,
        request_id: str,
        session_id: str,
    ) -> AsyncIterator[EventEnvelope]:
        """
        the single place where Strands agent is prepared and invoked.
        """
        with bind_observability_context(identity, request_id):
            context, agent = await self._prepare(
                identity, request_id=request_id, session_id=session_id
            )

            sequence = 0
            source = EventSource(
                agent_name=agent.name or self._factory._config.agent.name,
                invocation_id=context.request_id,
                kind="root",
            )

            tool_calls = self._tool_names_for(context)
            result = None

            try:
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
                    terminal: TurnOutcome = TurnInterrupt(
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

    async def stream_run(
        self,
        input_text: str,
        identity: IdentityContext,
        *,
        request_id: str,
        session_id: str,
    ) -> AsyncIterator[EventEnvelope]:
        """
        stream the agent for one turn
        """
        async for event in self._stream_turn(
            _Query(input=input_text),
            identity,
            request_id=request_id,
            session_id=session_id,
        ):
            yield event

    async def stream_resume(
        self,
        session_id: str,
        decisions: dict[str, str],
        identity: IdentityContext,
        *,
        request_id: str,
    ) -> AsyncIterator[EventEnvelope]:
        """
        stream the agent for one turn (interrupt response)
        """
        responses = [
            decision_to_interrupt_response(interrupt_id, decision)
            for interrupt_id, decision in decisions.items()
        ]
        async for event in self._stream_turn(
            _Query(input=responses),
            identity,
            request_id=request_id,
            session_id=session_id,
        ):
            yield event

    async def _collect(
        self, events: AsyncIterator[EventEnvelope]
    ) -> CompleteResult:
        """
        collects the events from stream and returns it as a single result.
        meant to be used by APIs that return synchronous results and do not stream anything.
        """
        outcome: TurnOutcome | None = None
        async for envelope in events:
            if isinstance(envelope.event, (TurnComplete, TurnInterrupt, TurnError)):
                outcome = envelope.event

        if outcome is None:
            return CompleteResult(
                outcome=TurnError(
                    code="AgentExecutionError",
                    message="Agent stream ended without a terminal event",
                    retryable=False,
                )
            )

        return CompleteResult(outcome=outcome)

    async def sync_run(
        self,
        input_text: str,
        identity: IdentityContext,
        *,
        request_id: str,
        session_id: str,
    ) -> CompleteResult:
        """
        run the agent for one turn synchronously
        """
        return await self._collect(
            self.stream_run(
                input_text,
                identity,
                request_id=request_id,
                session_id=session_id,
            )
        )

    async def sync_resume(
        self,
        session_id: str,
        decisions: dict[str, str],
        identity: IdentityContext,
        *,
        request_id: str,
    ) -> CompleteResult:
        """
        run the agent for one turn synchronously (interrupt response)
        """
        return await self._collect(
            self.stream_resume(
                session_id,
                decisions,
                identity,
                request_id=request_id,
            )
        )
