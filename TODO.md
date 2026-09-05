# todo

## better testing (not sure where/how each get classified as tho)

- agent factory
    - RBAC enforcement (tools, mcp, and subagents are built according to role)
    - new agent per request (no risk of leaking/using unprivileged agent config)
- token vault
    - just basic testing that public functions work as expected
- session mngr
    - technically not needed since Strands' session manager works and is tested out of the box
    - BUT we have 1 custom scaffold - make sure session IDs are created as expected 
- config files
    - agent-from-config works
    - tools/mcp-subagents-from-config works
- auth
    - properly blocks tool execution until token is valid
    - properly bubbles up auth requests from tool -> agent and tool -> subagent -> agent
    - properly bubbles down auth responses from caller -> agent -> tool and caller -> agent -> subagent -> tool
- HITL
    - properly blocks tool execution until Y
    - properly bubbles up HITL requests from tool -> agent and tool -> subagent -> agent
    - properly bubbles down HITL responses from caller -> agent -> tool and caller -> agent -> subagent -> tool
- event mapping
    - idk basic testing that we properly map Strands -> internal and internal -> API contract
- adapters
    - idk, basic unit testing that HTTP endpoints work ?
- deadline hook
    - properly stops agent execution (INCLUDING subagents) on deadline reached
- general orchestration tests (requires mock Strands model)
