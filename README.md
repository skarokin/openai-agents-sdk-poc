# agentpoc

Reference architecture for a multi-protocol agent service, where a single agent runtime serves many protocol (HTTP, A2A, MCP, etc.) adapters, with shared API contracts for callers (only required for HTTP). Built on FastAPI and the OpenAI Agents SDK; intended as a template to adapt.

Built with the following difficult items treated as first-class:
1. Inbound authentication - not everyone gets the same tools. Built-in agent factory allows you to write config files that gate certain tools by role.
2. Outbound authentication - tools can be authed OBO a user. Built-in approval interrupts & guardrails allow the agent to pause, request user consent, store user token, and resume the tool call with their credential.
3. One agent over many protocols - as your agent grows in adoption, so does the expected API surface - you may need to support CLI callers, MCP callers, A2A callers, Slack webhooks, etc. Built-in agent factory & formatter work supports this easily.

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
| `common/` | Shared dependencies, Pydantic request/response <-> internal dataclass mapping, tool/subagent binding helpers. |
| `config/` | Agents and tools as YAML. |
| `controls/` | Agent execution controls - hooks, guardrails, interrupts, etc. |
| `core/` | Agent factory, internal dataclasses, observability setup, SDK event <-> internal dataclass mapping, session manager. |
| `formatters/` | Agent orchestrators - `CompleteFormatter` (final outcome) or `EventsFormatter` (streamed results). | 
| `protocols/` | Protocol adapters (REST, SSE, stubs for MCP/A2A). Parse requests, read trusted identity, call a formatter. |
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

`--roles` changes which tools the factory exposes (RBAC demo). Session history and interrupted run checkpoints default to `.agent-data/`; set `AGENT_DATA_DIR` to override.

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
