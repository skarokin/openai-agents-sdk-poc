"""JSON session persistence tests."""

import tempfile
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

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


@pytest.mark.asyncio
async def test_save_run_state_recreates_run_state_dir():
    with tempfile.TemporaryDirectory() as directory:
        manager = SessionManager(Path(directory))
        run_state_dir = Path(directory) / "run_states"
        run_state_dir.rmdir()

        state = MagicMock()
        state.to_json.return_value = "{}"

        token = await manager.save_run_state(
            state,
            session_id="session-1",
            owner_subject_id="owner",
        )
        assert run_state_dir.is_dir()
        assert (run_state_dir / f"{token}.json").is_file()
