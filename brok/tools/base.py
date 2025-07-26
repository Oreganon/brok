"""Base classes for the tool system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from brok.exceptions import BrokError


class ToolExecutionError(BrokError):
    """Raised when a tool fails to execute."""


@dataclass
class ToolExecutionResult:
    """Result of a tool execution."""

    success: bool
    data: str
    error: str | None = None
    metadata: dict[str, Any] | None = None

    def to_text(self) -> str:
        """Convert result to a text representation for the LLM."""
        if self.success:
            return self.data
        else:
            return f"Error: {self.error or 'Unknown error occurred'}"


class BaseTool(ABC):
    """Abstract base class for all tools.

    Tools provide external functionality to the chatbot, such as accessing
    APIs for weather, calculations, time queries, etc.

    Example:
        >>> class WeatherTool(BaseTool):
        ...     name = "weather"
        ...     description = "Get current weather for a city"
        ...     parameters = {
        ...         "type": "object",
        ...         "properties": {
        ...             "city": {"type": "string", "description": "City name"}
        ...         },
        ...         "required": ["city"]
        ...     }
        ...
        ...     async def execute(self, **kwargs) -> ToolExecutionResult:
        ...         city = kwargs.get("city", "")
        ...         # API call logic here
        ...         return ToolExecutionResult(
        ...             success=True,
        ...             data=f"Current weather in {city}: Sunny, 22°C"
        ...         )
    """

    # Tool metadata - must be defined by subclasses
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    parameters: ClassVar[dict[str, Any]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate tool metadata when subclassing."""
        super().__init_subclass__(**kwargs)

        if not cls.name:
            raise ValueError(f"Tool {cls.__name__} must define a 'name' attribute")
        if not cls.description:
            raise ValueError(
                f"Tool {cls.__name__} must define a 'description' attribute"
            )
        if not isinstance(cls.parameters, dict):
            raise TypeError(f"Tool {cls.__name__} must define 'parameters' as a dict")

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolExecutionResult:
        """Execute the tool with the given parameters.

        Args:
            **kwargs: Tool parameters as defined in the parameters schema

        Returns:
            ToolExecutionResult: The result of the tool execution

        Raises:
            ToolExecutionError: If the tool execution fails
        """

    def validate_parameters(self, params: dict[str, Any]) -> None:
        """Validate parameters against the tool's schema.

        Args:
            params: Parameters to validate

        Raises:
            ToolExecutionError: If parameters are invalid
        """
        # Basic validation - check required parameters
        required = self.parameters.get("required", [])
        properties = self.parameters.get("properties", {})

        for param in required:
            if param not in params:
                raise ToolExecutionError(f"Missing required parameter: {param}")

        # Type validation for provided parameters
        for param, value in params.items():
            if param in properties:
                expected_type = properties[param].get("type")
                if expected_type == "string" and not isinstance(value, str):
                    raise ToolExecutionError(f"Parameter '{param}' must be a string")
                elif expected_type == "number" and not isinstance(value, int | float):
                    raise ToolExecutionError(f"Parameter '{param}' must be a number")
                elif expected_type == "boolean" and not isinstance(value, bool):
                    raise ToolExecutionError(f"Parameter '{param}' must be a boolean")

    def get_schema(self) -> dict[str, Any]:
        """Get the JSON schema for this tool.

        Returns:
            dict: JSON schema describing the tool and its parameters
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def __str__(self) -> str:
        """String representation of the tool."""
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        """Detailed string representation of the tool."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"description='{self.description[:50]}{'...' if len(self.description) > 50 else ''}')"
        )
