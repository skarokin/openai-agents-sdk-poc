"""Local stdio MCP server with two notes tools."""

import sys

from mcp.server.fastmcp import FastMCP
from strands.tools.mcp import MCPClient

notes = FastMCP("notes")

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
    """Delete a demo note by id. Requires human approval via HumanInTheLoop."""
    text = NOTES.get(note_id)
    if text is None:
        return f"No note with id {note_id}."
    return f"Deleted note {note_id}: {text}"


def create_notes_mcp(*, tool_filters=None) -> MCPClient:
    """Strands MCP client for the local notes stdio server."""

    from mcp import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agent.tools.notes_mcp"],
    )
    return MCPClient(
        lambda: stdio_client(params),
        tool_filters=tool_filters,
        prefix=None,
    )


if __name__ == "__main__":
    notes.run(transport="stdio")
