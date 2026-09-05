"""Built-in demonstration tools."""

import ast
import logging
import operator
from collections.abc import Callable

from strands import tool
from strands.types.tools import ToolContext

from agent.controls.interrupts import identity_from_tool_context, require_vault_token

logger = logging.getLogger(__name__)

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("Exponent magnitude cannot exceed 10")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("Expression contains unsupported syntax")


@tool(context=True)
def calculator(tool_context: ToolContext, expression: str) -> str:
    """Evaluate a basic arithmetic expression.

    Args:
        expression: Arithmetic using numbers, parentheses, +, -, *, /, //, %, or **.
    """

    if len(expression) > 200:
        raise ValueError("Expression is too long")
    identity = identity_from_tool_context(tool_context)
    logger.info("calculator invoked", extra={"subject_id": identity.subject_id})
    result = _evaluate(ast.parse(expression, mode="eval"))
    return f"{result:g}"


@tool(context=True)
def approval_demo(tool_context: ToolContext) -> str:
    """Demo HITL approval. Gated by HumanInTheLoop intervention before execution."""

    identity = identity_from_tool_context(tool_context)
    logger.info("approval demo tool invoked", extra={"subject_id": identity.subject_id})
    return "tool approved!"


@tool(context=True)
async def authentication_demo(tool_context: ToolContext) -> str:
    """Demo auth interrupt raised inside the tool (separate from HITL)."""

    access_token = await require_vault_token(
        tool_context,
        service="authentication_demo",
        tool_name="authentication_demo",
    )
    identity = identity_from_tool_context(tool_context)
    logger.info(
        "authentication demo tool invoked",
        extra={"subject_id": identity.subject_id, "access_token": access_token},
    )
    return "at this point, user has authenticated successfully!"


@tool(context=True)
async def auth_approval_demo(tool_context: ToolContext) -> str:
    """HITL via HumanInTheLoop (config hitl: true); auth via vault interrupt here."""

    access_token = await require_vault_token(
        tool_context,
        service="auth_approval_demo",
        tool_name="auth_approval_demo",
    )
    identity = identity_from_tool_context(tool_context)
    logger.info(
        "auth approval demo tool invoked",
        extra={"subject_id": identity.subject_id, "access_token": access_token},
    )
    return (
        "at this point, user has authenticated successfully "
        "and the tool has been approved!"
    )
