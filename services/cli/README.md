# Agent CLI

Interactive client for the POC agent service.

```powershell
uv run agent-cli --stream
uv run agent-cli "Calculate (12 + 8) * 3"
```

The CLI sends sample trusted-edge identity headers with the `calculator`,
`approval_user`, and `subagent_user` roles. Override them with `--subject`,
`--actor`, and `--roles`.
