"""Shared Pydantic contracts for the agent HTTP APIs."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SUBJECT_HEADER = "X-Subject-ID"
ACTOR_HEADER = "X-Actor-ID"
ROLES_HEADER = "X-Roles"


class ContractModel(BaseModel):
    """Base contract with strict field handling."""

    model_config = ConfigDict(extra="forbid")


class IdentityHeaders(ContractModel):
    """Trusted identity values supplied by the edge authorizer."""

    subject_id: str = Field(min_length=1)
    actor_id: str | None = None
    roles: frozenset[str] = Field(default_factory=frozenset)

    def to_http_headers(self) -> dict[str, str]:
        headers = {
            SUBJECT_HEADER: self.subject_id,
            ROLES_HEADER: ",".join(sorted(self.roles)),
        }
        if self.actor_id:
            headers[ACTOR_HEADER] = self.actor_id
        return headers


class AgentRunRequest(ContractModel):
    """Start a new agent turn."""

    input: str = Field(min_length=1)
    session_id: str | None = Field(default=None, min_length=1)


class ApprovalDecision(ContractModel):
    """Approve or reject one pending tool invocation."""

    interruption_id: str = Field(min_length=1)
    decision: Literal["approve", "reject"]


class AgentResumeRequest(ContractModel):
    """Resume an interrupted agent turn via the same session_id."""

    session_id: str = Field(min_length=1)
    decisions: tuple[ApprovalDecision, ...] = Field(min_length=1)


class Usage(ContractModel):
    requests: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ApprovalRequest(ContractModel):
    interruption_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    agent_name: str | None = None
    requires_auth: bool = False
    requires_approval: bool = True
    authenticated: bool = False
    service: str | None = None
    authorization_url: str | None = None


class AuthRequiredInfo(ContractModel):
    call_id: str
    tool_name: str
    service: str
    authorization_url: str


class VaultTokenRequest(ContractModel):
    service: str = Field(min_length=1)
    access_token: str = Field(min_length=1)


class VaultTokenResponse(ContractModel):
    ok: Literal[True] = True
    service: str


class CompletedResponse(ContractModel):
    status: Literal["completed"] = "completed"
    request_id: str
    session_id: str
    output: Any
    usage: Usage
    auth_required: tuple[AuthRequiredInfo, ...] = ()


class InterruptedResponse(ContractModel):
    status: Literal["interrupted"] = "interrupted"
    request_id: str
    session_id: str
    interruptions: tuple[ApprovalRequest, ...]


class ErrorResponse(ContractModel):
    status: Literal["error"] = "error"
    request_id: str
    session_id: str | None = None
    code: str
    message: str
    retryable: bool = False


AgentResponse = Annotated[
    CompletedResponse | InterruptedResponse | ErrorResponse,
    Field(discriminator="status"),
]


class EventSource(ContractModel):
    agent_name: str
    invocation_id: str
    parent_invocation_id: str | None = None
    kind: Literal["root", "handoff", "agent_tool", "a2a", "mcp"] = "root"


class StreamEvent(ContractModel):
    """One normalized event sent over SSE."""

    sequence: int = Field(ge=1)
    request_id: str
    session_id: str
    source: EventSource
    event_type: Literal[
        "text_chunk",
        "reasoning_chunk",
        "tool_start",
        "tool_result",
        "guardrail_tripped",
        "annotation",
        "auth_required",
        "usage_update",
        "agent_changed",
        "turn_complete",
        "turn_interrupt",
        "turn_error",
    ]
    data: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(ContractModel):
    status: Literal["ok"] = "ok"
    service: str = "agent"
