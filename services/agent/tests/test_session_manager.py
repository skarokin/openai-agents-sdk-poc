"""JSON session persistence tests."""

import tempfile
from pathlib import Path
from typing import Any, cast

import pytest

from agent.core.session_manager import SessionManager


@pytest.mark.asyncio
async def test_json_session_round_trip():
    with tempfile.TemporaryDirectory() as directory:
        manager = SessionManager(Path(directory))
        session = manager.session("example/session", "session-owner")
        await session.add_items(
            [
                {
                    "role": "user",
                    "content": "hello",
                }
            ]
        )

        stored = cast(dict[str, Any], (await session.get_items())[0])
        assert stored["role"] == "user"
        popped = await session.pop_item()
        assert popped is not None
        assert await session.get_items() == []


@pytest.mark.asyncio
async def test_sessions_are_isolated_by_owner():
    with tempfile.TemporaryDirectory() as directory:
        manager = SessionManager(Path(directory))
        owner_session = manager.session("shared", "owner")
        await owner_session.add_items([{"role": "user", "content": "hello"}])
        other_session = manager.session("shared", "intruder")
        assert await other_session.get_items() == []
        stored = await owner_session.get_items()
        assert stored[0]["role"] == "user"
