"""Base classes for the tool system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING, Any, ClassVar

from brok.exceptions import BrokError

if TYPE_CHECKING:
    from brok.tools.cache import Cache


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

    Supports optional caching for expensive operations like API calls.

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
        ...     def __init__(self):
        ...         super().__init__(cache_ttl_seconds=600)  # Cache for 10 minutes
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

    def __init__(
        self,
        cache: Cache[ToolExecutionResult] | None = None,
        cache_ttl_seconds: float | None = None,
    ) -> None:
        """Initialize the tool with optional caching.

        Args:
            cache: Cache instance to use for storing results
            cache_ttl_seconds: Time to live for cached results in seconds
        """
        self._cache = cache
        self._cache_ttl_seconds = cache_ttl_seconds

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

    async def execute_cached(self, **kwargs: Any) -> ToolExecutionResult:
        """Execute the tool with caching support.

        This is the main entry point for tool execution. It handles caching
        automatically if configured.

        Args:
            **kwargs: Tool parameters as defined in the parameters schema

        Returns:
            ToolExecutionResult: The result of the tool execution

        Raises:
            ToolExecutionError: If the tool execution fails
        """
        # Validate parameters first
        self.validate_parameters(kwargs)

        # Try cache if available
        if self._cache:
            cache_key = self._generate_cache_key(kwargs)
            cached_result = await self._cache.get(cache_key)
            if cached_result:
                # Update metadata to indicate cache hit
                if cached_result.metadata:
                    cached_result.metadata["cache_hit"] = True
                else:
                    cached_result.metadata = {"cache_hit": True}
                return cached_result

        # Execute the tool
        result = await self.execute(**kwargs)

        # Cache successful results if cache is configured
        if self._cache and result.success:
            cache_key = self._generate_cache_key(kwargs)
            await self._cache.set(cache_key, result, self._cache_ttl_seconds)

            # Update metadata to indicate cache miss
            if result.metadata:
                result.metadata["cache_hit"] = False
            else:
                result.metadata = {"cache_hit": False}

        return result

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolExecutionResult:
        """Execute the tool with the given parameters.

        This method must be implemented by subclasses. It should contain the
        actual tool logic without worrying about caching (use execute_cached
        for cached execution).

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

    def _generate_cache_key(self, params: dict[str, Any]) -> str:
        """Generate a cache key for the given parameters.

        Args:
            params: Tool parameters

        Returns:
            str: Cache key uniquely identifying this tool execution
        """
        # Sort parameters for consistent key generation
        sorted_params = dict(sorted(params.items()))

        # Create a hash of tool name + parameters
        key_data = {
            "tool": self.name,
            "params": sorted_params,
        }

        # Convert to JSON and create hash
        key_json = json.dumps(key_data, sort_keys=True, separators=(",", ":"))
        cache_key = hashlib.md5(key_json.encode()).hexdigest()

        return f"{self.name}:{cache_key}"

    def set_cache(
        self,
        cache: Cache[ToolExecutionResult],
        ttl_seconds: float | None = None,
    ) -> None:
        """Set the cache for this tool.

        Args:
            cache: Cache instance to use
            ttl_seconds: Time to live for cached results, overrides default
        """
        self._cache = cache
        if ttl_seconds is not None:
            self._cache_ttl_seconds = ttl_seconds

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics if caching is enabled.

        Returns:
            dict: Cache statistics or None if no cache is configured
        """
        if self._cache is not None and hasattr(self._cache, "get_stats"):
            return self._cache.get_stats()  # type: ignore[no-any-return]
        return None

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
