"""
Python interpreter tool for code execution.
"""

import sys
import io
import contextlib
from typing import Dict, Any
from .base import BaseTool, ToolResult


class PythonInterpreterTool(BaseTool):
    """Tool for executing Python code safely."""
    
    def __init__(self):
        super().__init__(
            name="python_interpreter",
            description="Executes Python code and returns the output",
            capabilities=["code_execution", "data_processing", "calculations"]
        )

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        """Execute Python code."""
        try:
            code = args.get("code", "")
            if not code:
                return ToolResult(
                    content=None,
                    success=False,
                    error_message="No code provided"
                )

            # Execute code and capture output
            output, error = self._safe_execute(code)
            
            if error:
                return ToolResult(
                    content=None,
                    success=False,
                    error_message=f"Execution error: {error}"
                )
            
            return ToolResult(
                content=output,
                success=True,
                confidence=0.9,
                metadata={"code": code, "output_length": len(output)}
            )
            
        except Exception as e:
            return ToolResult(
                content=None,
                success=False,
                error_message=f"Interpreter error: {str(e)}"
            )

    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for Python interpreter arguments."""
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        }

    def validate_args(self, args: Dict[str, Any]) -> bool:
        """Validate Python interpreter arguments."""
        return "code" in args and isinstance(args["code"], str)

    def _safe_execute(self, code: str) -> tuple[str, str]:
        """Safely execute Python code and return output and errors."""
        # Capture stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # Redirect output streams
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            # Create restricted execution environment
            restricted_globals = {
                "__builtins__": {
                    "abs": abs,
                    "all": all,
                    "any": any,
                    "bool": bool,
                    "dict": dict,
                    "enumerate": enumerate,
                    "filter": filter,
                    "float": float,
                    "int": int,
                    "len": len,
                    "list": list,
                    "map": map,
                    "max": max,
                    "min": min,
                    "pow": pow,
                    "print": print,
                    "range": range,
                    "round": round,
                    "set": set,
                    "sorted": sorted,
                    "str": str,
                    "sum": sum,
                    "tuple": tuple,
                    "type": type,
                    "zip": zip
                },
                "math": __import__("math"),
                "random": __import__("random"),
                "datetime": __import__("datetime"),
                "json": __import__("json"),
                "re": __import__("re")
            }
            
            restricted_locals = {}
            
            # Execute the code
            exec(code, restricted_globals, restricted_locals)
            
            output = stdout_capture.getvalue()
            error = stderr_capture.getvalue()
            
            return output, error
            
        except Exception as e:
            return "", str(e)
        finally:
            # Restore original streams
            sys.stdout = old_stdout
            sys.stderr = old_stderr
