"""
Subagent catalog: define real Agents here, wrap in SubagentSpec for YAML discovery.

as_subagent_tool (via the factory) injects non-negotiables: shared session, HITL,
deadline hook, and interrupt bubbling.
"""

import os

from strands import Agent

from agent.core.agent_factory import openai_model
from agent.core.models import SubagentSpec
from agent.tools.builtin import (
    approval_demo,
    auth_approval_demo,
    authentication_demo,
    calculator,
)

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

_SHARED_TOOLS = [
    calculator,
    approval_demo,
    authentication_demo,
    auth_approval_demo,
]


def _subagent(name: str) -> Agent:
    return Agent(
        name=name,
        description=SUBAGENT_DESCRIPTION,
        system_prompt=SUBAGENT_INSTRUCTIONS,
        model=openai_model(os.getenv("AGENT_MODEL", "gpt-5.6-luna")),
        tools=_SHARED_TOOLS,
        callback_handler=None,
    )


protected_subagent = SubagentSpec(agent=_subagent("Protected Subagent"))
open_subagent = SubagentSpec(agent=_subagent("Open Subagent"))
