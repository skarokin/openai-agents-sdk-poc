"""Yield one normalized stream across root and nested agent executions."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import suppress
from uuid import UUID

from agents import Agent, AgentUpdatedStreamEvent, Runner

from agent.core.agent_factory import AgentFactory
from agent.core.auth import TokenVault, auth_challenges_from_run
from agent.core.event_mapping import map_approvals, map_stream_event, map_usage
from agent.core.models import (
    AgentContext,
    AuthRequired,
    EventEnvelope,
    EventSink,
    EventSource,
    IdentityContext,
    Query,
    RunEvent,
    TurnComplete,
    TurnError,
    TurnInterrupt,
)
from agent.core.observability import bind_observability_context
from agent.core.runtime import create_agent_context, create_run_config
from agent.core.session_manager import SessionManager
from agent.hooks import GLOBAL_DEADLINE_HOOK

logger = logging.getLogger(__name__)
_CLOSED = object()


class QueueEventSink(EventSink):
    """Merge root, nested-agent, and application events in arrival order."""

    def __init__(self):
        self._queue: asyncio.Queue[EventEnvelope | object] = asyncio.Queue()
        self._sequence = 0
        self._lock = asyncio.Lock()
        self.auth_required: list[AuthRequired] = []

    async def emit(self, event: RunEvent, *, source: EventSource) -> None:
        async with self._lock:
            self._sequence += 1
            if isinstance(event, AuthRequired):
                self.auth_required.append(event)
            envelope = EventEnvelope(
                sequence=self._sequence,
                source=source,
                event=event,
            )
            await self._queue.put(envelope)

    async def close(self) -> None:
        await self._queue.put(_CLOSED)

    async def receive(self) -> EventEnvelope | None:
        item = await self._queue.get()
        return None if item is _CLOSED else item  # type: ignore[return-value]


class EventsFormatter:
    def __init__(
        self,
        factory: AgentFactory,
        sessions: SessionManager,
        token_vault: TokenVault,
    ):
        self._factory = factory
        self._sessions = sessions
        self._token_vault = token_vault

    async def _produce(
        self,
        query: Query,
        context: AgentContext,
        agent: Agent[AgentContext],
        sink: QueueEventSink,
    ) -> None:
        current_agent_name = agent.name
        streamed = None

        try:
            with bind_observability_context(
                context.identity,
                context.request_id,
            ):
                streamed = Runner.run_streamed(
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

                async for sdk_event in streamed.stream_events():
                    source = EventSource(
                        agent_name=current_agent_name,
                        invocation_id=context.request_id,
                        kind="root",
                    )

                    for event in map_stream_event(
                        sdk_event,
                        previous_agent_name=current_agent_name,
                    ):
                        await sink.emit(event, source=source)

                    if isinstance(sdk_event, AgentUpdatedStreamEvent):
                        current_agent_name = sdk_event.new_agent.name

                terminal_source = EventSource(
                    agent_name=current_agent_name,
                    invocation_id=context.request_id,
                    kind="root",
                )

                if streamed.interruptions:
                    state = streamed.to_state()
                    token = await self._sessions.save_run_state(
                        state,
                        session_id=context.session_id,
                        owner_subject_id=context.identity.subject_id,
                    )

                    terminal: RunEvent = TurnInterrupt(
                        interruptions=map_approvals(streamed.interruptions),
                        run_state=state,
                        resume_token=token,
                    )
                else:
                    terminal = TurnComplete(
                        output=streamed.final_output,
                        usage=map_usage(streamed.context_wrapper.usage),
                        auth_required=auth_challenges_from_run(
                            streamed,
                            emitted=tuple(sink.auth_required),
                        ),
                    )

                await sink.emit(terminal, source=terminal_source)

        except asyncio.CancelledError:
            if streamed is not None:
                streamed.cancel()

            raise
        except Exception as exc:
            logger.exception(
                "streaming agent run failed",
                extra={
                    "request_id": context.request_id,
                    "subject_id": context.identity.subject_id,
                    "actor_id": context.identity.actor_id or "-",
                },
            )
            await sink.emit(
                TurnError(
                    code=type(exc).__name__,
                    message="Agent execution failed",
                    retryable=False,
                ),
                source=EventSource(
                    agent_name=current_agent_name,
                    invocation_id=context.request_id,
                    kind="root",
                ),
            )
        finally:
            await sink.close()

    async def _stream_prepared(
        self,
        query: Query,
        context: AgentContext,
        agent: Agent[AgentContext],
        sink: QueueEventSink,
    ) -> AsyncIterator[EventEnvelope]:
        producer = asyncio.create_task(
            self._produce(query, context, agent, sink),
            name=f"agent-stream-{context.request_id}",
        )
        try:
            while event := await sink.receive():
                yield event
            await producer
        finally:
            if not producer.done():
                producer.cancel()
                with suppress(asyncio.CancelledError):
                    await producer

    async def stream(
        self,
        input_text: str,
        identity: IdentityContext,
        *,
        request_id: str,
        session_id: str,
    ) -> AsyncIterator[EventEnvelope]:
        """Stream an agent run"""

        sink = QueueEventSink()
        context = create_agent_context(
            identity,
            sink,
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
            sink,
        ):
            yield event

    async def resume(
        self,
        token: UUID,
        decisions: dict[str, str],
        identity: IdentityContext,
        *,
        request_id: str,
    ) -> AsyncIterator[tuple[EventEnvelope, str]]:
        """Resume an interrupted agent run"""

        async with self._sessions.lock_run_state(token):
            session_id = self._sessions.run_state_session_id(
                token,
                identity.subject_id,
            )
            sink = QueueEventSink()
            context = create_agent_context(
                identity,
                sink,
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
            self._sessions.apply_decisions(state, decisions)
            completed = False
            async for event in self._stream_prepared(
                Query(input=state),
                context,
                agent,
                sink,
            ):
                if isinstance(event.event, TurnComplete | TurnInterrupt):
                    completed = True
                yield event, session_id
            if completed:
                await self._sessions.delete_run_state(token)
