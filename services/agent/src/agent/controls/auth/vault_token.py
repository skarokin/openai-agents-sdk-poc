"""Helpers for raising auth interrupts from tools."""

from typing import Any

from strands.types.tools import ToolContext

from agent.core.models import IdentityContext


async def require_vault_token(
    tool_context: ToolContext,
    *,
    service: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> str:
    """
    Raises an auth interrupt until the vault has a valid token

    RuntimeError will propagate to the caller which should handle it gracefully
    """

    vault = tool_context.invocation_state.get("token_vault")
    if vault is None:
        raise RuntimeError("token_vault missing from invocation_state")

    identity = IdentityContext.from_tool_context(tool_context)
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
