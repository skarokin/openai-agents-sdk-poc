"""HTTP contract and endpoint smoke tests."""

from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

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


class FakeAgentRunner:
    async def sync_run(
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

    async def stream_run(
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


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        app.state.agent_runner = FakeAgentRunner()
        yield test_client


@pytest.fixture
def headers():
    return {
        "X-Subject-ID": "http-user",
        "X-Actor-ID": "gateway",
        "X-Roles": "calculator,approval_user",
    }


def test_rest_contract(client, headers):
    response = client.post(
        "/v1/agent/runs",
        headers=headers,
        json={"input": "hello", "session_id": "http-session"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["output"] == "http-user: hello"


def test_sse_contract(client, headers):
    response = client.post(
        "/v1/agent/runs/stream",
        headers=headers,
        json={"input": "hello", "session_id": "http-session"},
    )
    assert response.status_code == 200
    assert "event: text_chunk" in response.text
    assert "event: turn_complete" in response.text
    assert "http-user: hello" in response.text


def test_identity_is_required(client):
    response = client.post(
        "/v1/agent/runs",
        json={"input": "hello"},
    )
    assert response.status_code == 401


def test_vault_token_endpoint_stores_for_caller(client, headers):
    response = client.post(
        "/v1/agent/vault/tokens",
        headers=headers,
        json={
            "service": "authentication_demo",
            "access_token": "http-demo-token",
        },
    )
    assert response.status_code == 200
    stored = app.state.token_vault.peek(
        IdentityContext(subject_id="http-user"),
        "authentication_demo",
    )
    assert stored is not None
    assert stored.access_token == "http-demo-token"
