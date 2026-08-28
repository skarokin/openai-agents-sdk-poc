"""JSON session persistence tests."""

import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

from agent.core.session_manager import SessionManager


class SessionManagerTest(unittest.IsolatedAsyncioTestCase):
    async def test_json_session_round_trip(self):
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
            self.assertEqual(stored["role"], "user")
            popped = await session.pop_item()
            self.assertIsNotNone(popped)
            self.assertEqual(await session.get_items(), [])

    async def test_sessions_are_isolated_by_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SessionManager(Path(directory))
            owner_session = manager.session("shared", "owner")
            await owner_session.add_items([{"role": "user", "content": "hello"}])
            other_session = manager.session("shared", "intruder")
            self.assertEqual(await other_session.get_items(), [])
            stored = await owner_session.get_items()
            self.assertEqual(stored[0]["role"], "user")


if __name__ == "__main__":
    unittest.main()
