# todo

## better testing (not sure where/how each get classified as tho)

1. RBAC works (session isolation, agent factory)
2. token vault works (put get peek challenge)
3. agent-from-config works
4. HITL properly bubbles up from subagents, repsonse bubbles down to subagents
5. auth interrupt properly bubble up from subagents, response bubble down to subagents
6. subagents and root agent share the same session but not the same namespace
7. subagents have same invocation state as root (exlcuding root agent key)
8. HITL works in general (no means no, yes means yes)
9. auth interrupt blocks forever until valid token
10. strands -> internal -> http event mapping works as expected
11. individual protocol adapter tests (things like "test happy path" and "if no identity then 401" or "properly receive event stream")
12. individual tool-level units (later - just scaffold/template it)
13. tools/subagents/mcp-from-config works
14. deadline hook works
15. 