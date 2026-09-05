"""
Subagent catalog. Only needs to define the agent name, instructions,
descriptions, and tools.

Tools are decoupled from the subagent definition - they are defined and used
elsewhere. HITL for invoking the subagent itself is configured in tool_config.yaml
(``hitl: true`` on the subagent entry).
"""

from agent.core.models import SubagentSpec

SUBAGENT_INSTRUCTIONS = (
    "You are a focused subagent. Use the calculator for arithmetic. "
    "When explicitly asked to demonstrate approval, call approval_demo. "
    "When asked to demonstrate auth, call authentication_demo. "
    "Return a concise answer to the parent agent."
)
SUBAGENT_DESCRIPTION = (
    "Delegate a focused task to a subagent that can calculate and "
    "demonstrate nested approval / auth workflows."
)
SHARED_LEAF_TOOLS = ("calculator", "approval_demo", "authentication_demo")

protected_subagent = SubagentSpec(
    agent_name="Protected Subagent",
    instructions=SUBAGENT_INSTRUCTIONS,
    description=SUBAGENT_DESCRIPTION,
    tools=SHARED_LEAF_TOOLS,
)
open_subagent = SubagentSpec(
    agent_name="Open Subagent",
    instructions=SUBAGENT_INSTRUCTIONS,
    description=SUBAGENT_DESCRIPTION,
    tools=SHARED_LEAF_TOOLS,
)
