"""Built-in demonstration tools."""

import ast
import logging
import operator
from collections.abc import Callable

from agents import RunContextWrapper
from agents.decorators import tool

from agent.controls.guardrails import auth_guardrail, get_access_token
from agent.controls.interrupts import auth_required, hitl_auth_required
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
    """
    This tool is a demo of the approval workflow. It will return a confirmation after the caller approves it.
    """

    logger.info(
        "approval demo tool invoked",
        extra={"subject_id": context.context.identity.subject_id},
    )
    return "tool approved!"


@tool(
    needs_approval=auth_required("authentication_demo"),
    tool_input_guardrails=[auth_guardrail("authentication_demo")],
)
async def authentication_demo(context: RunContextWrapper[AgentContext]) -> str:
    """
    This tool is a demo of the authentication workflow. It will return a confirmation after the caller authenticates.
    """

    access_token = await get_access_token(context.context, "authentication_demo")

    logger.info(
        "authentication demo tool invoked",
        extra={
            "subject_id": context.context.identity.subject_id,
            "access_token": access_token,
        },
    )
    return "at this point, user has authenticated successfully!"


@tool(
    needs_approval=hitl_auth_required("auth_approval_demo"),
    tool_input_guardrails=[auth_guardrail("auth_approval_demo")],
)
async def auth_approval_demo(context: RunContextWrapper[AgentContext]) -> str:
    """
    This tool is a demo of the authentication and approval workflow. It will return a confirmation after the caller authenticates and approves it.
    """

    access_token = await get_access_token(context.context, "auth_approval_demo")

    logger.info(
        "auth approval demo tool invoked",
        extra={
            "subject_id": context.context.identity.subject_id,
            "access_token": access_token,
        },
    )
    return (
        "at this point, user has authenticated successfully "
        "and the tool has been approved!"
    )
