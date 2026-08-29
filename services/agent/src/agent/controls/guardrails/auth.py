"""Vault token access and server-side auth enforcement guardrails."""

from collections.abc import Mapping
from typing import Any

from agents import (
    ToolGuardrailFunctionOutput,
    ToolInputGuardrail,
    ToolInputGuardrailData,
)

from agent.core.models import (
    AgentContext,
    AuthRequired,
    EventSource,
)

from .vault import AuthChallenge

AUTH_REJECT_MESSAGE = (
    "The tool did not run because the caller is not authenticated. "
    "The user must complete authentication via the provided authorization URL "
    "before this tool can be executed. "
)


class AuthTokenMissingError(RuntimeError):
    def __init__(self, service: str):
        super().__init__(f"No access token for service {service!r}")
        self.service = service


async def get_access_token(context: AgentContext, service: str) -> str:
    token = await context.token_vault.get(context.identity, service)
    if token is None:
        raise AuthTokenMissingError(service)

    return token.access_token


def auth_guardrail(service: str) -> ToolInputGuardrail[AgentContext]:
    """
    Reject tool execution with a model-visible error when the vault has no token.

    All tools that require authentication must use this guardrail.
    """

    async def check(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
        context = data.context.context
        if not isinstance(context, AgentContext):
            return ToolGuardrailFunctionOutput.allow()

        if await context.token_vault.has_token(context.identity, service):
            return ToolGuardrailFunctionOutput.allow()

        # at this point, vault does not have a token - we need to send an AuthRequired event
        # the caller is responsible for receiving and parsing the AuthRequired event
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
            AUTH_REJECT_MESSAGE,
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


def auth_info_from_guardrail_output(
    output: ToolGuardrailFunctionOutput,
) -> Mapping[str, Any] | None:
    behavior = output.behavior
    if not isinstance(behavior, Mapping) or behavior.get("type") != "reject_content":
        return None
    info = output.output_info
    if isinstance(info, Mapping) and info.get("code") == "auth_required":
        return info
    return None
