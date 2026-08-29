"""Approval pauses for auth and HITL."""

from dataclasses import dataclass
from typing import Any

from agents import FunctionTool, RunContextWrapper, ToolApprovalItem

from agent.core.models import AgentContext


APPROVAL_POLICY_ATTR = "__agent_approval_policy__"


@dataclass(frozen=True, slots=True)
class ApprovalPolicy:
    requires_auth: bool
    requires_approval: bool
    vault_service: str | None = None


def _tag_policy(
    needs_approval: Any,
    policy: ApprovalPolicy,
) -> Any:
    """Attaches metadata to the needs_approval function so that when interrupt happens, client knows what to collect."""
    setattr(needs_approval, APPROVAL_POLICY_ATTR, policy)
    return needs_approval


def approval_policy_from_needs_approval(setting: Any) -> ApprovalPolicy:
    if setting is True:
        return ApprovalPolicy(requires_auth=False, requires_approval=True)
    if setting is False:
        return ApprovalPolicy(requires_auth=False, requires_approval=False)

    policy = getattr(setting, APPROVAL_POLICY_ATTR, None)
    if isinstance(policy, ApprovalPolicy):
        return policy

    return ApprovalPolicy(requires_auth=False, requires_approval=True)


def _function_tool_for_item(item: ToolApprovalItem) -> FunctionTool | None:
    agent = item.agent
    tool_name = item.name
    if agent is None or not tool_name:
        return None

    for tool in agent.tools:
        if isinstance(tool, FunctionTool) and tool.name == tool_name:
            return tool

    return None


def approval_policy_for_item(item: ToolApprovalItem) -> ApprovalPolicy:
    """
    Returns the approval policy for a given tool item.

    This is used by map_approvals; it retrieves the needs_approval callable for that tool which
    has been tagged with the approval policy previously (by auth_required or hitl_auth_required).

    This tag creates the approval policy (requires_auth, requires_approval, vault_service) to assist the client
    in deciding how to handle the interrupt - there are 4 modes:
    1. Auth prompt -> auto-resume
    2. Already authed -> auto-resume
    3. Auth prompt -> HITL approval -> auto-resume
    4. Already authed -> HITL approval -> auto-resume
    """
    tool = _function_tool_for_item(item)
    if tool is None:
        return ApprovalPolicy(requires_auth=False, requires_approval=True)

    return approval_policy_from_needs_approval(tool.needs_approval)


def auth_required(tool_name: str):
    """
    Pause until the vault has a token.

    Clients should present user with the auth URL and send a resume token when the user has authenticated.
    """

    async def needs_approval(
        ctx: RunContextWrapper[AgentContext],
        _args: dict[str, Any],
        _call_id: str,
    ) -> bool:
        agent_context = ctx.context
        if not isinstance(agent_context, AgentContext):
            return False

        return not await agent_context.token_vault.has_token(
            agent_context.identity,
            tool_name,
        )

    return _tag_policy(
        needs_approval,
        ApprovalPolicy(
            requires_auth=True,
            requires_approval=False,
            vault_service=tool_name,
        ),
    )


def hitl_auth_required(tool_name: str):
    """
    Pause until the caller approves AND the vault has a token.

    Clients should present user with the auth URL and...
    (1) Collect human approval
    (2) Wait for user to authenticate
    before sending a resume token.

    The auth guardrail still rejects execution if the user approves, but no token exists.
    """

    async def needs_approval(
        ctx: RunContextWrapper[AgentContext],
        _args: dict[str, Any],
        call_id: str,
    ) -> bool:
        agent_context = ctx.context
        if not isinstance(agent_context, AgentContext):
            return False

        # if the tools hasn't been approved, interrupt
        if ctx.is_tool_approved(tool_name, call_id) is not True:
            return True

        # if the vault does not have a token, interrupt
        return not await agent_context.token_vault.has_token(
            agent_context.identity,
            tool_name,
        )

    return _tag_policy(
        needs_approval,
        ApprovalPolicy(
            requires_auth=True,
            requires_approval=True,
            vault_service=tool_name,
        ),
    )
