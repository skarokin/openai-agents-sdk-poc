"""Shared pytest fixtures and helpers for agent workflow tests."""

from __future__ import annotations

import time

import pytest

from agent.config import ServiceConfig, load_service_config
from agent.core.models import AgentContext, IdentityContext
from agent.core.token_vault import TokenVault


@pytest.fixture
def service_config() -> ServiceConfig:
    return load_service_config()


def make_context(
    *,
    subject_id: str = "test-user",
    roles: frozenset[str] | set[str] | None = None,
    request_id: str = "test-request",
    session_id: str = "test-session",
    deadline_offset_seconds: float = 60,
    token_vault: TokenVault | None = None,
    **kwargs,
) -> AgentContext:
    return AgentContext(
        identity=IdentityContext(
            subject_id=subject_id,
            roles=frozenset(roles or ()),
        ),
        request_id=request_id,
        session_id=session_id,
        deadline_epoch_seconds=time.time() + deadline_offset_seconds,
        token_vault=token_vault or TokenVault.from_environment(),
        **kwargs,
    )
