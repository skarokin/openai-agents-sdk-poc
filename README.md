# agentpoc

Reference architecture for a multi-protocol agent service, where a single agent runtime serves many protocol (HTTP, A2A, MCP, etc.) adapters, with shared API contracts for callers (only required for HTTP). Built on FastAPI and the AWS Strands SDK; intended as a template to adapt.

Built with the following difficult items treated as first-class:
1. Inbound authentication - not everyone gets the same tools. Built-in agent factory allows you to write config files that gate certain tools by role.
2. Outbound authentication - tools can be authed OBO a user. Auth pauses are raised as Strands interrupts inside tools; HITL uses `HumanInTheLoop` interventions. The agent pauses, requests consent/token, stores it in the vault, and resumes.
3. One agent over many protocols - as your agent grows in adoption, so does the expected API surface - you may need to support CLI callers, MCP callers, A2A callers, Slack webhooks, etc. Built-in agent factory & runner work supports this easily.

Authentication assumes a trusted upstream source provides identity!

## Layout

```
packages/agent-common/   Shared Pydantic HTTP contracts (runs, interrupts, auth requests, SSE events, etc.)
services/agent/          FastAPI service
services/cli/            REST/SSE client (handles HITL + auth demos)
```

Inside `services/agent/src/agent/`:

| Path | Role |
|------|------|
| `common/` | Shared dependencies, Pydantic request/response <-> internal dataclass mapping. |
| `config/` | Agents and tools as YAML. |
| `controls/` | Agent execution controls - deadlines, HITL, auth. |
| `core/` | Agent factory, internal dataclasses, observability setup, Strands event mapping, FileSessionManager helpers. |
| `runners/` | `AgentRunner` - one streaming turn path; `sync_run`/`sync_resume` collect the terminal result. |
| `endpoints/` | Caller-facing entry points (REST, SSE, stubs for MCP/A2A). Parse requests, read trusted identity, call the runner. |
| `tools/` | Builtin tools, subagents, MCP client wiring. |

## Run

```bash
uv sync --all-packages
export OPENAI_API_KEY=...
uv run agent
```

Another terminal:

```bash
uv run agent-cli --stream
```

`--roles` changes which tools the factory exposes (RBAC demo). Conversation and interrupt state live in Strands `FileSessionManager` under `.agent-data/sessions/`; set `AGENT_DATA_DIR` to override.

## API

- `POST /v1/agent/runs` — run to completion
- `POST /v1/agent/runs/resume` — resume after interrupt
- `POST /v1/agent/runs/stream` — SSE event stream
- `POST /v1/agent/runs/stream/resume` — resume streaming run
- `POST /v1/agent/vault/tokens` — store token for user
- `GET /health`

## Tests

```powershell
uv run pytest services/agent/tests -q
```
