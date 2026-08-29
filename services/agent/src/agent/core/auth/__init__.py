"""Authentication helpers."""

from .auth_approval import (
    ApprovalPolicy,
    approval_policy_for_item,
    approval_policy_from_needs_approval,
    auth_required,
    hitl_auth_required,
)
from .auth_guardrail import (
    AUTH_REJECT_MESSAGE,
    AuthTokenMissingError,
    auth_guardrail,
    auth_info_from_guardrail_output,
    get_access_token,
)
from .token_vault import AuthChallenge, StoredToken, TokenVault

__all__ = [
    "AUTH_REJECT_MESSAGE",
    "ApprovalPolicy",
    "AuthChallenge",
    "AuthTokenMissingError",
    "StoredToken",
    "TokenVault",
    "approval_policy_for_item",
    "approval_policy_from_needs_approval",
    "auth_guardrail",
    "auth_info_from_guardrail_output",
    "auth_required",
    "get_access_token",
    "hitl_auth_required",
]
