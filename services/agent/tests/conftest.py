"""Shared pytest fixtures and helpers for agent workflow tests."""

from __future__ import annotations

import time

import pytest

from agent.config import ServiceConfig, load_service_config
from agent.core.models import AgentContext, EventSource, IdentityContext, RunEvent


class CollectingSink:
    def __init__(self) -> None:
        self.events: list[tuple[RunEvent, EventSource]] = []

    async def emit(self, event: RunEvent, *, source: EventSource) -> None:
        self.events.append((event, source))


@pytest.fixture
def service_config() -> ServiceConfig:
    return load_service_config()


@pytest.fixture
def collecting_sink() -> CollectingSink:
    return CollectingSink()


def make_context(
    *,
    subject_id: str = "test-user",
    roles: frozenset[str] | set[str] | None = None,
    event_sink: CollectingSink | None = None,
    request_id: str = "test-request",
    session_id: str = "test-session",
    deadline_offset_seconds: float = 60,
    **kwargs,
) -> AgentContext:
    return AgentContext(
        identity=IdentityContext(
            subject_id=subject_id,
            roles=frozenset(roles or ()),
        ),
        event_sink=event_sink or CollectingSink(),
        request_id=request_id,
        session_id=session_id,
        deadline_epoch_seconds=time.time() + deadline_offset_seconds,
        **kwargs,
    )
