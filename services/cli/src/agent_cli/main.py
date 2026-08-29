"""Interactive REST/SSE client with HITL approval handling."""

import argparse
import json
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import httpx
from agent_common import (
    AgentResponse,
    AgentResumeRequest,
    AgentRunRequest,
    ApprovalDecision,
    IdentityHeaders,
    InterruptedResponse,
    StreamEvent,
)
from pydantic import TypeAdapter

_RESPONSE_ADAPTER = TypeAdapter(AgentResponse)
_DEFAULT_ROLES = "calculator,approval_user,subagent_user,mcp_user"


def _render(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, indent=2)


def _decisions(interruptions: list[dict[str, Any]]) -> tuple[ApprovalDecision, ...]:
    decisions: list[ApprovalDecision] = []
    for interruption in interruptions:
        tool = interruption.get("tool_name", "unknown_tool")
        agent = interruption.get("agent_name") or "unknown agent"
        arguments = _render(interruption.get("arguments", {}))
        answer = input(
            f"\nApprove {tool} from {agent} with arguments {arguments}? [y/N] "
        )
        decisions.append(
            ApprovalDecision(
                interruption_id=interruption["interruption_id"],
                decision="approve"
                if answer.strip().lower() in {"y", "yes"}
                else "reject",
            )
        )
    return tuple(decisions)


class AgentClient:
    def __init__(self, base_url: str, identity: IdentityHeaders):
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=identity.to_http_headers(),
            timeout=None,
        )
        self._last_text_source: tuple[str, str] | None = None

    def close(self) -> None:
        self._client.close()

    def _post(self, path: str, body: dict[str, Any]):
        response = self._client.post(path, json=body)
        response.raise_for_status()
        return _RESPONSE_ADAPTER.validate_python(response.json())

    def run_rest(self, prompt: str, session_id: str):
        response = self._post(
            "/v1/agent/runs",
            AgentRunRequest(input=prompt, session_id=session_id).model_dump(
                mode="json"
            ),
        )
        while isinstance(response, InterruptedResponse):
            decisions = _decisions(
                [item.model_dump(mode="json") for item in response.interruptions]
            )
            response = self._post(
                "/v1/agent/runs/resume",
                AgentResumeRequest(
                    resume_token=response.resume_token,
                    decisions=decisions,
                ).model_dump(mode="json"),
            )
        if response.status == "completed":
            print(_render(response.output))
            print(
                f"[usage: {response.usage.total_tokens} tokens, "
                f"{response.usage.requests} requests]"
            )
        else:
            print(f"[error: {response.code}] {response.message}")

    def _sse(
        self,
        path: str,
        body: dict[str, Any],
    ) -> Iterator[StreamEvent]:
        with self._client.stream("POST", path, json=body) as response:
            response.raise_for_status()
            data_lines: list[str] = []
            for line in response.iter_lines():
                if not line:
                    if data_lines:
                        yield StreamEvent.model_validate_json("\n".join(data_lines))
                        data_lines.clear()
                    continue
                if line.startswith("data:"):
                    data_lines.append(line.removeprefix("data:").lstrip())
            if data_lines:
                yield StreamEvent.model_validate_json("\n".join(data_lines))

    def _display_event(self, event: StreamEvent) -> None:
        source = event.source.agent_name
        data = event.data
        if event.event_type == "text_chunk":
            source_key = (event.source.kind, source)
            if source_key != self._last_text_source and event.source.kind != "root":
                print(f"\n[{source}] ", end="")
            self._last_text_source = source_key
            print(data.get("delta", ""), end="", flush=True)
        elif event.event_type == "reasoning_chunk":
            print(f"\n[{source} reasoning] {data.get('delta', '')}", end="")
        elif event.event_type == "tool_start":
            print(f"\n[{source} → {data.get('tool_name')}]")
        elif event.event_type == "tool_result":
            print(
                f"\n[{source} ← {data.get('tool_name')}: {_render(data.get('output'))}]"
            )
        elif event.event_type == "agent_changed":
            print(f"\n[agent changed → {data.get('agent_name')}]")
        elif event.event_type == "guardrail_tripped":
            print(f"\n[guardrail: {data.get('message')}]")
        elif event.event_type == "turn_error":
            print(f"\n[error: {data.get('code')}] {data.get('message')}")

    def run_sse(self, prompt: str, session_id: str) -> None:
        self._last_text_source = None
        path = "/v1/agent/runs/stream"
        body: dict[str, Any] = AgentRunRequest(
            input=prompt,
            session_id=session_id,
        ).model_dump(mode="json")
        while True:
            interrupted: dict[str, Any] | None = None
            for event in self._sse(path, body):
                self._display_event(event)
                if event.event_type == "turn_interrupt":
                    interrupted = event.data
                elif event.event_type == "turn_complete":
                    usage = event.data.get("usage", {})
                    print(
                        f"\n[usage: {usage.get('total_tokens', 0)} tokens, "
                        f"{usage.get('requests', 0)} requests]"
                    )
            if interrupted is None:
                return
            body = AgentResumeRequest(
                resume_token=interrupted["resume_token"],
                decisions=_decisions(interrupted["interruptions"]),
            ).model_dump(mode="json")
            path = "/v1/agent/runs/stream/resume"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--subject", default="poc-user")
    parser.add_argument("--actor")
    parser.add_argument("--roles", default=_DEFAULT_ROLES)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("prompt", nargs="?")
    return parser


def main() -> None:
    args = _parser().parse_args()
    identity = IdentityHeaders(
        subject_id=args.subject,
        actor_id=args.actor,
        roles=frozenset(role.strip() for role in args.roles.split(",") if role.strip()),
    )
    client = AgentClient(args.base_url, identity)
    session_id = uuid4().hex
    try:
        prompt = args.prompt
        while True:
            if prompt is None:
                prompt = input("\nYou (or 'quit'): ").strip()
            if not prompt or prompt.lower() in {"quit", "exit"}:
                return
            if args.stream:
                client.run_sse(prompt, session_id)
            else:
                client.run_rest(prompt, session_id)
            prompt = None
    finally:
        client.close()


if __name__ == "__main__":
    main()
