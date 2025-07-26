"""Tool registry for managing available tools."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from brok.tools.base import BaseTool, ToolExecutionError

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing and accessing available tools.

    The registry maintains a collection of tools and provides methods to
    discover, validate, and execute them.

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register_tool(WeatherTool())
        >>> tools = registry.get_available_tools()
        >>> result = await registry.execute_tool("weather", {"city": "London"})
    """

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self._tools: dict[str, BaseTool] = {}
        logger.debug("Initialized tool registry")

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool in the registry.

        Args:
            tool: The tool instance to register

        Raises:
            ValueError: If a tool with the same name is already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")

        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def unregister_tool(self, name: str) -> None:
        """Unregister a tool from the registry.

        Args:
            name: Name of the tool to unregister

        Raises:
            KeyError: If the tool is not registered
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")

        del self._tools[name]
        logger.info(f"Unregistered tool: {name}")

    def get_tool(self, name: str) -> BaseTool:
        """Get a tool by name.

        Args:
            name: Name of the tool to retrieve

        Returns:
            BaseTool: The requested tool

        Raises:
            KeyError: If the tool is not registered
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")

        return self._tools[name]

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: Name of the tool to check

        Returns:
            bool: True if the tool is registered, False otherwise
        """
        return name in self._tools

    def get_available_tools(self) -> list[str]:
        """Get a list of all available tool names.

        Returns:
            list[str]: List of registered tool names
        """
        return list(self._tools.keys())

    def get_tools_schema(self) -> list[dict[str, Any]]:
        """Get the JSON schema for all registered tools.

        Returns:
            list[dict]: List of tool schemas for LLM prompt
        """
        return [tool.get_schema() for tool in self._tools.values()]

    async def execute_tool(self, name: str, parameters: dict[str, Any]) -> str:
        """Execute a tool with the given parameters.

        Args:
            name: Name of the tool to execute
            parameters: Parameters to pass to the tool

        Returns:
            str: Result of the tool execution as text

        Raises:
            KeyError: If the tool is not registered
            ToolExecutionError: If the tool execution fails
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")

        tool = self._tools[name]

        # Validate parameters
        try:
            tool.validate_parameters(parameters)
        except ToolExecutionError as e:
            logger.warning(f"Parameter validation failed for tool '{name}': {e}")
            raise

        # Execute tool
        try:
            logger.debug(f"Executing tool '{name}' with parameters: {parameters}")
            result = await tool.execute(**parameters)

            if result.success:
                logger.info(f"Tool '{name}' executed successfully")
            else:
                logger.warning(f"Tool '{name}' execution failed: {result.error}")

            return result.to_text()

        except Exception as e:
            logger.exception(f"Unexpected error executing tool '{name}'")
            raise ToolExecutionError(f"Tool execution failed: {e}") from e

    def get_tools_description(self) -> str:
        """Get a human-readable description of all available tools.

        Returns:
            str: Description of available tools for LLM prompt
        """
        if not self._tools:
            return "No tools available."

        descriptions = []
        descriptions.append("Available tools:")

        for tool in self._tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")

            # Add parameter information
            properties = tool.parameters.get("properties", {})
            required = tool.parameters.get("required", [])

            if properties:
                params = []
                for param, schema in properties.items():
                    param_desc = schema.get("description", "")
                    param_type = schema.get("type", "unknown")
                    is_required = param in required

                    param_str = f"{param} ({param_type})"
                    if is_required:
                        param_str += " [required]"
                    if param_desc:
                        param_str += f": {param_desc}"

                    params.append(param_str)

                descriptions.append(f"  Parameters: {', '.join(params)}")

        return "\\n".join(descriptions)

    def __len__(self) -> int:
        """Get the number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered using 'in' operator."""
        return name in self._tools

    def __iter__(self) -> Iterator[BaseTool]:
        """Iterate over registered tools."""
        return iter(self._tools.values())
