"""HTTP contract and protocol adapter smoke tests."""

import unittest
from collections.abc import AsyncIterator

from agent.core.models import (
    CompleteResult,
    EventEnvelope,
    EventSource,
    IdentityContext,
    TextChunk,
    TurnComplete,
    UsageUpdate,
)
from agent.main import app
from fastapi.testclient import TestClient


class FakeCompleteFormatter:
    async def run(
        self,
        input_text: str,
        identity: IdentityContext,
        *,
        request_id: str,
        session_id: str,
    ) -> CompleteResult:
        return CompleteResult(
            TurnComplete(
                output=f"{identity.subject_id}: {input_text}",
                usage=UsageUpdate(requests=1, total_tokens=3),
            )
        )


class FakeEventsFormatter:
    async def stream(
        self,
        input_text: str,
        identity: IdentityContext,
        *,
        request_id: str,
        session_id: str,
    ) -> AsyncIterator[EventEnvelope]:
        source = EventSource(
            agent_name="fake",
            invocation_id=request_id,
        )
        yield EventEnvelope(
            sequence=1,
            source=source,
            event=TextChunk(delta=f"{identity.subject_id}: {input_text}"),
        )
        yield EventEnvelope(
            sequence=2,
            source=source,
            event=TurnComplete(
                output="done",
                usage=UsageUpdate(requests=1, total_tokens=3),
            ),
        )


class HttpProtocolsTest(unittest.TestCase):
    def setUp(self):
        self._client_context = TestClient(app)
        self.client = self._client_context.__enter__()
        app.state.complete_formatter = FakeCompleteFormatter()
        app.state.events_formatter = FakeEventsFormatter()
        self.headers = {
            "X-Subject-ID": "http-user",
            "X-Actor-ID": "gateway",
            "X-Roles": "calculator,approval_user",
        }

    def tearDown(self):
        self._client_context.__exit__(None, None, None)

    def test_rest_contract(self):
        response = self.client.post(
            "/v1/agent/runs",
            headers=self.headers,
            json={"input": "hello", "session_id": "http-session"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "completed")
        self.assertEqual(response.json()["output"], "http-user: hello")

    def test_sse_contract(self):
        response = self.client.post(
            "/v1/agent/runs/stream",
            headers=self.headers,
            json={"input": "hello", "session_id": "http-session"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("event: text_chunk", response.text)
        self.assertIn("event: turn_complete", response.text)
        self.assertIn("http-user: hello", response.text)

    def test_identity_is_required(self):
        response = self.client.post(
            "/v1/agent/runs",
            json={"input": "hello"},
        )
        self.assertEqual(response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
