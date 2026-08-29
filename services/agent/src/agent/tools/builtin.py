"""Built-in demonstration tools."""

import ast
import logging
import operator
from collections.abc import Callable

from agents import RunContextWrapper
from agents.decorators import tool

from agent.core.models import AgentContext

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


@tool
async def calculator(
    context: RunContextWrapper[AgentContext],
    expression: str,
) -> str:
    """Evaluate a basic arithmetic expression.

    Args:
        expression: Arithmetic using numbers, parentheses, +, -, *, /, //, %, or **.
    """

    if len(expression) > 200:
        raise ValueError("Expression is too long")
    logger.info(
        "calculator invoked",
        extra={"subject_id": context.context.identity.subject_id},
    )
    result = _evaluate(ast.parse(expression, mode="eval"))
    return f"{result:g}"


@tool(needs_approval=True)
async def approval_demo(context: RunContextWrapper[AgentContext]) -> str:
    """Return a confirmation after the caller approves this tool."""

    logger.info(
        "approval demo tool invoked",
        extra={"subject_id": context.context.identity.subject_id},
    )
    return "tool approved!"
