"""Vault-backed auth checks as tool input guardrails."""

from collections.abc import Mapping
from typing import Any

from agents import (
    ToolGuardrailFunctionOutput,
    ToolInputGuardrail,
    ToolInputGuardrailData,
)
from agents.tool_guardrails import ToolInputGuardrailResult

from agent.core.models import (
    AgentContext,
    AuthRequired,
    EventSink,
    EventSource,
    RunEvent,
)

from .token_vault import AuthChallenge

MODEL_AUTH_MESSAGE = (
    "The user has been asked to authenticate into the service. The tool did not run."
)


class AuthTokenMissingError(RuntimeError):
    def __init__(self, service: str):
        super().__init__(f"No access token for service {service!r}")
        self.service = service


async def get_access_token(context: AgentContext, service: str) -> str:
    token = await context.token_vault.get(context.identity, service)
    if token is None:
        # we raise here because if get_access_token is called, we expect a token to be present
        # this is because the execution flow is:
        #   tool called -> (optional) HITL approval -> auth guardrail -> tool execution
        # where auth guardrail -> tool execution can only happen if and only if the token is present & valid
        raise AuthTokenMissingError(service)

    return token.access_token


def auth_guardrail(service: str) -> ToolInputGuardrail[AgentContext]:
    """
    Return a guardrail that blocks the tool when the vault has no token.

    Runs after HITL approval. On miss, rejects with structured output_info and emits AuthRequired
    """

    async def check(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
        context = data.context.context
        if not isinstance(context, AgentContext):
            return ToolGuardrailFunctionOutput.allow()

        if await context.token_vault.has_token(context.identity, service):
            return ToolGuardrailFunctionOutput.allow()

        challenge = context.token_vault.challenge(service)
        payload = _auth_payload(
            challenge=challenge,
            tool_name=data.context.tool_name,
            call_id=data.context.tool_call_id,
        )

        await context.event_sink.emit(
            AuthRequired(
                call_id=data.context.tool_call_id,
                tool_name=data.context.tool_name,
                service=challenge.service,
                authorization_url=challenge.authorization_url,
            ),
            source=EventSource(
                agent_name=data.agent.name if data.agent is not None else "unknown",
                invocation_id=context.request_id,
                kind="root",
            ),
        )
        return ToolGuardrailFunctionOutput.reject_content(
            MODEL_AUTH_MESSAGE,
            output_info=payload,
        )

    return ToolInputGuardrail(guardrail_function=check, name=f"auth_{service}")


def _auth_payload(
    *,
    challenge: AuthChallenge,
    tool_name: str,
    call_id: str,
) -> dict[str, str]:
    return {
        "code": "auth_required",
        "service": challenge.service,
        "authorization_url": challenge.authorization_url,
        "tool_name": tool_name,
        "call_id": call_id,
    }


def auth_challenges_from_guardrail_results(
    results: list[ToolInputGuardrailResult],
) -> tuple[AuthRequired, ...]:
    """Collect auth challenges from tool input guardrail rejections."""

    challenges: list[AuthRequired] = []
    for result in results:
        payload = _auth_info(result.output)
        if payload is None:
            continue
        challenges.append(_auth_required_from_payload(payload))
    return tuple(challenges)


def auth_challenges_from_run(
    result: Any,
    *,
    emitted: tuple[AuthRequired, ...] = (),
) -> tuple[AuthRequired, ...]:
    from_guardrails = auth_challenges_from_guardrail_results(
        result.tool_input_guardrail_results,
    )
    return _merge_auth_challenges(from_guardrails, emitted)


def _merge_auth_challenges(
    *groups: tuple[AuthRequired, ...],
) -> tuple[AuthRequired, ...]:
    merged: list[AuthRequired] = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        for item in group:
            key = (item.call_id, item.service)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return tuple(merged)


class AuthCollectingSink:
    """
    Capture AuthRequired events while delegating to another sink.

    A new sink is required because auth is signaled via an event, not part of tool result or agent run result.
    The complete fformatter doesn't stream events to the client, so we need something to capture these events
    until the turn is complete.
    """

    def __init__(self, inner: EventSink):
        self._inner = inner
        self.auth_required: list[AuthRequired] = []

    async def emit(self, event: RunEvent, *, source: EventSource) -> None:
        if isinstance(event, AuthRequired):
            self.auth_required.append(event)
        await self._inner.emit(event, source=source)

    @property
    def collected(self) -> tuple[AuthRequired, ...]:
        return tuple(self.auth_required)


def _auth_required_from_payload(payload: Mapping[str, Any]) -> AuthRequired:
    return AuthRequired(
        call_id=str(payload.get("call_id", "")),
        tool_name=str(payload.get("tool_name", "unknown_tool")),
        service=str(payload["service"]),
        authorization_url=str(payload["authorization_url"]),
    )


def _auth_info(output: ToolGuardrailFunctionOutput) -> Mapping[str, Any] | None:
    behavior = output.behavior
    if not isinstance(behavior, Mapping) or behavior.get("type") != "reject_content":
        return None
    info = output.output_info
    if isinstance(info, Mapping) and info.get("code") == "auth_required":
        return info
    return None
