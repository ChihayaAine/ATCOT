"""
Calculator tool for mathematical computations.
"""

import math
import re
from typing import Dict, Any
from .base import BaseTool, ToolResult


class CalculatorTool(BaseTool):
    """Tool for performing mathematical calculations."""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Performs mathematical calculations and evaluates expressions",
            capabilities=["arithmetic", "mathematical_functions", "expression_evaluation"]
        )

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        """Execute mathematical calculation."""
        try:
            expression = args.get("expression", "")
            if not expression:
                return ToolResult(
                    content=None,
                    success=False,
                    error_message="No expression provided"
                )

            # Sanitize and evaluate expression
            sanitized_expr = self._sanitize_expression(expression)
            result = self._safe_eval(sanitized_expr)
            
            return ToolResult(
                content=result,
                success=True,
                confidence=0.95,
                metadata={"original_expression": expression, "sanitized": sanitized_expr}
            )
            
        except Exception as e:
            return ToolResult(
                content=None,
                success=False,
                error_message=f"Calculation error: {str(e)}"
            )

    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for calculator arguments."""
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }

    def validate_args(self, args: Dict[str, Any]) -> bool:
        """Validate calculator arguments."""
        return "expression" in args and isinstance(args["expression"], str)

    def _sanitize_expression(self, expression: str) -> str:
        """Sanitize mathematical expression for safe evaluation."""
        # Remove any potentially dangerous operations
        expression = re.sub(r'[^0-9+\-*/.()%\s]', '', expression)
        
        # Replace common mathematical functions
        expression = expression.replace("^", "**")
        
        return expression.strip()

    def _safe_eval(self, expression: str) -> float:
        """Safely evaluate mathematical expression."""
        # Define allowed names for evaluation
        allowed_names = {
            "__builtins__": {},
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e
        }
        
        return eval(expression, allowed_names, {})
