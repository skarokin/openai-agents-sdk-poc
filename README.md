# Multi-protocol agent POC

This uv workspace contains:

- `services/agent`: FastAPI/OpenAI Agents SDK service
- `services/cli`: interactive REST and SSE client
- `packages/agent-common`: shared Pydantic HTTP contracts

## Run

```powershell
uv sync --all-packages
$env:OPENAI_API_KEY = "..."
uv run agent
```

In another terminal:

```powershell
uv run agent-cli --stream
```

Use `--roles` to inspect RBAC behavior. The default CLI roles grant all POC
tools. Identity headers are trusted only because the service assumes an
external gateway performs authentication and overwrites them.

Session history and interrupted `RunState` checkpoints are stored under
`.agent-data/` by default. Set `AGENT_DATA_DIR` to change the location.
