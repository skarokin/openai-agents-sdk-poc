"""Helpers for raising auth interrupts from tools."""

from typing import Any

from strands.types.tools import ToolContext

from agent.core.models import IdentityContext


def identity_from_tool_context(tool_context: ToolContext) -> IdentityContext:
    raw = tool_context.agent.state.get("identity") or {}
    invocation_identity = tool_context.invocation_state.get("identity")
    if isinstance(invocation_identity, IdentityContext):
        return invocation_identity
    return IdentityContext(
        subject_id=str(raw.get("subject_id") or ""),
        actor_id=raw.get("actor_id"),
        roles=frozenset(raw.get("roles") or ()),
    )


async def require_vault_token(
    tool_context: ToolContext,
    *,
    service: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> str:
    """Raise an auth interrupt until the vault has a token, then return it.

    HITL is separate: gate approval with ``HumanInTheLoop``, not this helper.
    """

    vault = tool_context.invocation_state.get("token_vault")
    if vault is None:
        raise RuntimeError("token_vault missing from invocation_state")

    identity = identity_from_tool_context(tool_context)
    token = vault.peek(identity, service)
    if token is not None:
        return token.access_token

    challenge = vault.challenge(service)
    tool_context.interrupt(
        f"auth-{service}",
        reason={
            "requires_auth": True,
            "requires_approval": False,
            "service": challenge.service,
            "authorization_url": challenge.authorization_url,
            "tool_name": tool_name,
            "arguments": arguments or {},
        },
    )

    stored = await vault.get(identity, service)
    if stored is None:
        raise RuntimeError(
            f"No vault token for service '{service}' after auth interrupt"
        )
    return stored.access_token
