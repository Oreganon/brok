"""Calculator tool for mathematical calculations."""

from __future__ import annotations

import ast
import logging
import math
import operator
import re
from typing import Any, ClassVar

from brok.tools.base import BaseTool, ToolExecutionResult

logger = logging.getLogger(__name__)


class CalculatorTool(BaseTool):
    """Tool for performing mathematical calculations.

    Safely evaluates mathematical expressions using Python's AST module.
    Supports basic arithmetic, trigonometry, and common math functions.

    Example:
        >>> tool = CalculatorTool()
        >>> result = await tool.execute(expression="2 + 3 * 4")
        >>> print(result.data)  # "2 + 3 * 4 = 14"
    """

    name: ClassVar[str] = "calculator"
    description: ClassVar[str] = (
        "Perform mathematical calculations and evaluate expressions"
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4', 'sqrt(16)', 'sin(pi/2)')",
            }
        },
        "required": ["expression"],
    }

    # Allowed operators and functions for safe evaluation
    OPERATORS: ClassVar[dict] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    FUNCTIONS: ClassVar[dict] = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "ceil": math.ceil,
        "floor": math.floor,
        "factorial": math.factorial,
    }

    CONSTANTS: ClassVar[dict] = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
    }

    def __init__(self) -> None:
        """Initialize the calculator tool."""
        super().__init__()

    async def execute(self, **kwargs: Any) -> ToolExecutionResult:
        """Execute the calculator tool to evaluate an expression.

        Args:
            expression: Mathematical expression to evaluate

        Returns:
            ToolExecutionResult: Calculation result or error
        """
        expression = kwargs.get("expression", "").strip()

        if not expression:
            return ToolExecutionResult(
                success=False, data="", error="Expression is required"
            )

        try:
            # Clean and normalize the expression
            cleaned_expr = self._clean_expression(expression)

            # Parse and evaluate safely
            result = self._safe_eval(cleaned_expr)

            # Format the result
            formatted_result = self._format_result(cleaned_expr, result)

            return ToolExecutionResult(
                success=True,
                data=formatted_result,
                metadata={
                    "expression": expression,
                    "cleaned_expression": cleaned_expr,
                    "result": result,
                    "result_type": type(result).__name__,
                },
            )

        except ZeroDivisionError:
            return ToolExecutionResult(success=False, data="", error="Division by zero")
        except OverflowError:
            return ToolExecutionResult(
                success=False, data="", error="Result too large to calculate"
            )
        except ValueError as e:
            return ToolExecutionResult(
                success=False, data="", error=f"Invalid calculation: {e!s}"
            )
        except Exception as e:
            logger.exception(f"Error evaluating expression: {expression}")
            return ToolExecutionResult(
                success=False, data="", error=f"Calculation error: {e!s}"
            )

    def _clean_expression(self, expression: str) -> str:
        """Clean and normalize the mathematical expression."""
        # Remove common text and normalize
        expression = expression.lower()

        # Replace common text patterns
        replacements = [
            ("what is", ""),
            ("calculate", ""),
            ("compute", ""),
            ("solve", ""),
            ("equals", ""),
            ("=", ""),
            ("*", "*"),
            ("÷", "/"),
            ("π", "pi"),
        ]

        for old, new in replacements:
            expression = expression.replace(old, new)

        # Handle implicit multiplication (e.g., "2pi" -> "2*pi")

        expression = re.sub(r"(\d+)([a-z])", r"\1*\2", expression)
        expression = re.sub(r"([a-z])(\d+)", r"\1*\2", expression)
        expression = re.sub(r"\)(\d)", r")*\1", expression)
        expression = re.sub(r"(\d)\(", r"\1*(", expression)

        return expression.strip()

    def _safe_eval(self, expression: str) -> float | int | list[float | int]:
        """Safely evaluate a mathematical expression using AST."""
        try:
            # Parse the expression
            tree = ast.parse(expression, mode="eval")

            # Evaluate the AST
            return self._eval_node(tree.body)

        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {e!s}") from e

    def _eval_node(self, node: ast.AST) -> float | int | list[float | int]:
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int | float):
                return node.value
            else:
                raise TypeError(f"Unsupported constant type: {type(node.value)}")

        elif isinstance(node, ast.Name):
            if node.id in self.CONSTANTS:
                value = self.CONSTANTS[node.id]
                if isinstance(value, int | float):
                    return value
                else:
                    raise ValueError(f"Constant {node.id} is not a number")
            else:
                raise ValueError(f"Unknown variable: {node.id}")

        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.OPERATORS.get(type(node.op))
            if op:
                return op(left, right)  # type: ignore[no-any-return]
            else:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")

        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.OPERATORS.get(type(node.op))
            if op:
                return op(operand)  # type: ignore[no-any-return]
            else:
                raise ValueError(
                    f"Unsupported unary operator: {type(node.op).__name__}"
                )

        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name in self.FUNCTIONS:
                args = [self._eval_node(arg) for arg in node.args]
                return self.FUNCTIONS[func_name](*args)  # type: ignore[no-any-return]
            else:
                raise ValueError(f"Unknown function: {func_name}")

        elif isinstance(node, ast.List):
            # For lists, ensure all elements are numbers, not nested lists
            elements = []
            for item in node.elts:
                result = self._eval_node(item)
                if isinstance(result, int | float):
                    elements.append(result)
                else:
                    raise TypeError(
                        f"List elements must be numbers, got {type(result).__name__}"
                    )
            return elements

        else:
            raise TypeError(f"Unsupported expression type: {type(node).__name__}")

    def _format_result(
        self, expression: str, result: float | int | list[float | int]
    ) -> str:
        """Format the calculation result for display."""
        if isinstance(result, list):
            return f"{expression} = {result}"

        # Round floating point results to avoid long decimals
        if isinstance(result, float):
            if result.is_integer():
                result = int(result)
            else:
                # Round to 10 decimal places to avoid floating point precision issues
                result = round(result, 10)
                # Remove trailing zeros
                if isinstance(result, float) and str(result).endswith(".0"):
                    result = int(result)

        return f"{expression} = {result}"
