# Agent service

FastAPI service built on OpenAI Agents SDK 0.22.

Set `OPENAI_API_KEY`, then run:

```powershell
uv run agent
```

Endpoints:

- `POST /v1/agent/runs`
- `POST /v1/agent/runs/resume`
- `POST /v1/agent/runs/stream`
- `POST /v1/agent/runs/stream/resume`
- `GET /health`

The POC trusts identity headers supplied by an external authorizer:
`X-Subject-ID`, optional `X-Actor-ID`, and comma-separated `X-Roles`. Do not
expose it directly without a gateway that authenticates and overwrites these
headers.
