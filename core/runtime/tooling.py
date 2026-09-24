"""Tool abstraction, signature schema inspection, and soft schema validation."""

import asyncio
import inspect
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, get_args, get_origin
import jsonschema

from core.workspace.session import SessionWorkspace


class ToolSchemaValidator:
    """Soft validator for tool invocation arguments against JSON schema definitions."""

    @staticmethod
    def validate(schema: Dict[str, Any], arguments: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate input arguments against a JSON schema without throwing unhandled exceptions.

        Args:
            schema: Valid JSON schema dictionary describing expected parameters.
            arguments: Actual arguments dictionary supplied by the LLM.

        Returns:
            Tuple[bool, List[str]]: (is_valid, list of human-readable violation descriptions).
        """
        if not schema:
            return True, []

        validator = jsonschema.Draft7Validator(schema)
        errors = list(validator.iter_errors(arguments))

        if not errors:
            return True, []

        violations: List[str] = []
        for err in errors:
            path = " -> ".join(str(p) for p in err.absolute_path)
            prefix = f"Field '{path}': " if path else ""
            violations.append(f"{prefix}{err.message}")

        return False, violations


class RuntimeTool:
    """Encapsulates a callable tool executable within the agent runtime loop.

    Supports automatic parameter schema extraction from Python type signatures,
    soft schema validation with violation reporting, and sync/async invocation.

    Attributes:
        name: Unique tool name for function calling.
        description: Informative explanation of tool capability for the model.
        func: Backing callable (sync or async).
        param_schema: JSON Schema describing expected parameters.
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        param_schema: Dict[str, Any],
    ) -> None:
        self.name = name
        self.description = description
        self.func = func
        self.param_schema = param_schema

    @classmethod
    def from_callable(
        cls,
        name: str,
        description: str,
        func: Callable[..., Any],
        param_schema: Optional[Dict[str, Any]] = None,
    ) -> "RuntimeTool":
        """Construct a RuntimeTool from a Python callable, automatically inferring schema if omitted.

        Args:
            name: Identifier for the tool.
            description: Description of the tool's purpose and usage.
            func: Target callable function or coroutine.
            param_schema: Optional explicit parameter JSON schema.

        Returns:
            RuntimeTool: Configured tool instance.
        """
        if param_schema is None:
            param_schema = cls._infer_param_schema(func)

        return cls(
            name=name,
            description=description,
            func=func,
            param_schema=param_schema,
        )

    @staticmethod
    def _map_python_type_to_json_schema(py_type: Any) -> str:
        """Map standard Python types to JSON schema types."""
        origin = get_origin(py_type)
        if origin is Union:
            args = [a for a in get_args(py_type) if a is not type(None)]
            if args:
                return RuntimeTool._map_python_type_to_json_schema(args[0])
            return "string"

        if py_type in (str, Optional[str]):
            return "string"
        if py_type in (int, Optional[int]):
            return "integer"
        if py_type in (float, Optional[float]):
            return "number"
        if py_type in (bool, Optional[bool]):
            return "boolean"
        if py_type in (list, List) or origin in (list, List):
            return "array"
        if py_type in (dict, Dict) or origin in (dict, Dict):
            return "object"
        return "string"

    @classmethod
    def _infer_param_schema(cls, func: Callable[..., Any]) -> Dict[str, Any]:
        """Inspect callable signature and annotations to generate JSON schema."""
        sig = inspect.signature(func)
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param_name, param in sig.parameters.items():
            # Skip framework-injected context variables
            if param_name in ("workspace", "self", "cls"):
                continue

            prop_schema: Dict[str, Any] = {}
            if param.annotation != inspect.Parameter.empty:
                prop_schema["type"] = cls._map_python_type_to_json_schema(param.annotation)
            else:
                prop_schema["type"] = "string"

            if param.default == inspect.Parameter.empty:
                required.append(param_name)
            else:
                prop_schema["default"] = param.default

            properties[param_name] = prop_schema

        schema: Dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required

        return schema

    def validate_arguments(self, arguments: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate input arguments against tool parameter schema.

        Args:
            arguments: Dictionary of arguments supplied by LLM.

        Returns:
            Tuple[bool, List[str]]: (is_valid, violation_errors).
        """
        return ToolSchemaValidator.validate(self.param_schema, arguments)

    async def execute(
        self,
        arguments: Union[Dict[str, Any], str],
        workspace: Optional[SessionWorkspace] = None,
    ) -> Any:
        """Execute the backing callable with provided arguments, handling sync or async functions.

        Args:
            arguments: Parsed arguments dict or raw JSON string.
            workspace: Optional active session workspace for file or storage access.

        Returns:
            Any: Tool execution output result.
        """
        if arguments is None:
            args_dict = {}
        elif isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                args_dict = parsed if isinstance(parsed, dict) else {"input": parsed}
            except json.JSONDecodeError:
                args_dict = {"input": arguments}
        elif isinstance(arguments, dict):
            args_dict = dict(arguments)
        else:
            args_dict = {}

        # Inject workspace if requested in function signature
        sig = inspect.signature(self.func)
        if "workspace" in sig.parameters and "workspace" not in args_dict:
            args_dict["workspace"] = workspace

        if inspect.iscoroutinefunction(self.func):
            return await self.func(**args_dict)
        return await asyncio.to_thread(self.func, **args_dict)

    def to_openai_format(self) -> Dict[str, Any]:
        """Export tool declaration to OpenAI tool calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.param_schema,
            },
        }
