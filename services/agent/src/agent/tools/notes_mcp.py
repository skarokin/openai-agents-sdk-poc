"""Local stdio MCP server with two tools. HITL is applied by the Agents SDK client."""

from mcp.server.mcpserver import MCPServer

notes = MCPServer("notes")

NOTES = {
    "1": "buy milk",
    "2": "call sam",
}


@notes.tool()
def list_notes() -> str:
    """List the demo notes. Safe; does not require approval."""
    return "\n".join(f"{note_id}. {text}" for note_id, text in NOTES.items())


@notes.tool()
def delete_note(note_id: str) -> str:
    """Delete a demo note by id. Requires human approval."""
    text = NOTES.get(note_id)
    if text is None:
        return f"No note with id {note_id}."
    return f"Deleted note {note_id}: {text}"


def create_notes_mcp(*, tool_filter=None):
    """SDK client. HITL is on this instantiation; RBAC comes from tool_config.yaml."""
    import sys

    from agents.mcp import MCPServerStdio

    return MCPServerStdio(
        name="notes",
        cache_tools_list=True,
        params={
            "command": sys.executable,
            "args": ["-m", "agent.tools.notes_mcp"],
        },
        require_approval={"always": {"tool_names": ["delete_note"]}},
        tool_filter=tool_filter,
    )


if __name__ == "__main__":
    notes.run()
