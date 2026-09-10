"""Interactive REST/SSE client with HITL approval handling."""

import argparse
import json
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import httpx
from pydantic import TypeAdapter

from agent_common import (
    AgentResponse,
    AgentResumeRequest,
    AgentRunRequest,
    ApprovalDecision,
    IdentityHeaders,
    InterruptedResponse,
    StreamEvent,
    VaultTokenRequest,
)

_RESPONSE_ADAPTER = TypeAdapter(AgentResponse)
_DEFAULT_ROLES = "calculator,approval_user,subagent_user,mcp_user,auth_user"


def _render(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, indent=2)


def _format_tool_start(source: str, tool_name: str, arguments: Any) -> str:
    """Shared display for root and nested (subagent) tool starts."""

    return f"[{source} → {tool_name}: {_render(arguments)}]"


def _format_tool_result(source: str, tool_name: str, output: Any) -> str:
    """Shared display for root and nested (subagent) tool results."""

    return f"[{source} ← {tool_name}: {_render(output)}]"


def _prompt_auth(*, tool_name: str, service: str, authorization_url: str) -> None:
    print(f"\nAuthentication required for {tool_name} ({service}).")
    print(f"Open: {authorization_url}")
    print("Continue after you have authorized externally.")
    input("Press Enter when ready to store a demo token and resume...")


class AgentClient:
    def __init__(self, base_url: str, identity: IdentityHeaders):
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=identity.to_http_headers(),
            timeout=None,
        )
        self._last_text_source: tuple[str, str] | None = None
        # current_tool_use carries the name; toolResult only has toolUseId.
        self._tool_names: dict[str, str] = {}

    def close(self) -> None:
        self._client.close()

    def _store_demo_token(self, service: str) -> None:
        response = self._client.post(
            "/v1/agent/vault/tokens",
            json=VaultTokenRequest(
                service=service,
                access_token=f"demo-{uuid4().hex}",
            ).model_dump(mode="json"),
        )
        response.raise_for_status()

    def _decisions(
        self, interruptions: list[dict[str, Any]]
    ) -> tuple[ApprovalDecision, ...]:
        decisions: list[ApprovalDecision] = []
        for interruption in interruptions:
            interruption_id = interruption["interruption_id"]
            tool = interruption.get("tool_name", "unknown_tool")
            requires_auth = interruption.get("requires_auth", False)
            requires_approval = interruption.get("requires_approval", True)
            authenticated = interruption.get("authenticated", False)
            service = interruption.get("service")
            authorization_url = interruption.get("authorization_url")

            just_authenticated = False
            if requires_auth and not authenticated:
                if not service or not authorization_url:
                    raise ValueError(
                        f"Auth metadata missing for {tool} interruption"
                    )
                _prompt_auth(
                    tool_name=tool,
                    service=service,
                    authorization_url=authorization_url,
                )
                self._store_demo_token(service)
                just_authenticated = True

            if requires_approval:
                agent = interruption.get("agent_name") or "unknown agent"
                arguments = _render(interruption.get("arguments", {}))
                if requires_auth and authenticated and not just_authenticated:
                    prompt = (
                        f"\nAlready authenticated. Approve {tool} from {agent} "
                        f"with arguments {arguments}? [y/N] "
                    )
                else:
                    prompt = (
                        f"\nApprove {tool} from {agent} with arguments "
                        f"{arguments}? [y/N] "
                    )
                answer = input(prompt)
                decision = (
                    "approve"
                    if answer.strip().lower() in {"y", "yes"}
                    else "reject"
                )
            else:
                decision = "approve"

            decisions.append(
                ApprovalDecision(
                    interruption_id=interruption_id,
                    decision=decision,
                )
            )
        return tuple(decisions)

    def _post(self, path: str, body: dict[str, Any]):
        response = self._client.post(path, json=body)
        response.raise_for_status()
        return _RESPONSE_ADAPTER.validate_python(response.json())

    def run_rest(self, prompt: str, session_id: str):
        while True:
            response = self._post(
                "/v1/agent/runs",
                AgentRunRequest(input=prompt, session_id=session_id).model_dump(
                    mode="json"
                ),
            )
            while isinstance(response, InterruptedResponse):
                response = self._post(
                    "/v1/agent/runs/resume",
                    AgentResumeRequest(
                        session_id=response.session_id,
                        decisions=self._decisions(
                            [
                                item.model_dump(mode="json")
                                for item in response.interruptions
                            ]
                        ),
                    ).model_dump(mode="json"),
                )
            if response.status == "error":
                print(f"[error: {response.code}] {response.message}")
                return
            print(_render(response.output))
            print(
                f"[usage: {response.usage.total_tokens} tokens, "
                f"{response.usage.requests} requests]"
            )
            return

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
            call_id = str(data.get("call_id") or "")
            tool_name = str(data.get("tool_name") or "")
            if call_id and tool_name:
                self._tool_names[call_id] = tool_name

            print(
                "\n"
                + _format_tool_start(
                    source,
                    tool_name or call_id or "tool",
                    data.get("arguments", {}),
                )
            )
        elif event.event_type == "tool_result":
            call_id = str(data.get("call_id") or "")
            tool_name = (
                self._tool_names.get(call_id)
                or data.get("tool_name")
                or call_id
                or "tool"
            )
            print(
                "\n"
                + _format_tool_result(source, str(tool_name), data.get("output"))
            )
        elif event.event_type == "auth_required":
            print(
                f"\n[auth required: {data.get('tool_name')} → {data.get('authorization_url')}]"
            )
        elif event.event_type == "agent_changed":
            print(f"\n[agent changed → {data.get('agent_name')}]")
        elif event.event_type == "guardrail_tripped":
            print(f"\n[guardrail: {data.get('message')}]")
        elif event.event_type == "annotation":
            print(f"\n[annotation:{data.get('kind')}] {data.get('data')}")
        elif event.event_type == "turn_error":
            print(f"\n[error: {data.get('code')}] {data.get('message')}")
        elif event.event_type == "turn_complete":
            self._tool_names.clear()

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
            if interrupted is not None:
                body = AgentResumeRequest(
                    session_id=session_id,
                    decisions=self._decisions(interrupted["interruptions"]),
                ).model_dump(mode="json")
                path = "/v1/agent/runs/stream/resume"
                continue
            return


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
