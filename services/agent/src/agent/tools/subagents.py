"""
Subagent catalog. Only needs to define the agent name, instructions, descriptions, and tools.

Tools are decoupled from the subagent definition - they are defined and used elsewhere.
"""

from agent.core.models import SubagentSpec

SUBAGENT_INSTRUCTIONS = (
    "You are a focused subagent. Use the calculator for arithmetic. "
    "When explicitly asked to demonstrate approval, call approval_demo. "
    "Return a concise answer to the parent agent."
)
SUBAGENT_DESCRIPTION = (
    "Delegate a focused task to a subagent that can calculate and "
    "demonstrate nested approval workflows."
)
SHARED_LEAF_TOOLS = ("calculator", "approval_demo")

protected_subagent = SubagentSpec(
    agent_name="Protected Subagent",
    instructions=SUBAGENT_INSTRUCTIONS,
    description=SUBAGENT_DESCRIPTION,
    tools=SHARED_LEAF_TOOLS,
    needs_approval=True,
)
open_subagent = SubagentSpec(
    agent_name="Open Subagent",
    instructions=SUBAGENT_INSTRUCTIONS,
    description=SUBAGENT_DESCRIPTION,
    tools=SHARED_LEAF_TOOLS,
)
