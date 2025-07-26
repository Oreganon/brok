"""Tool system for brok chatbot.

This module provides a framework for extending the bot with tools that can
access external APIs and services to provide real-time information.

Available tools:
- Weather: Get current weather and forecasts
- Calculator: Perform mathematical calculations
- Time: Time and date utilities

Example:
    >>> from brok.tools import ToolRegistry
    >>> registry = ToolRegistry()
    >>> tools = registry.get_available_tools()
    >>> weather_tool = registry.get_tool("weather")
    >>> result = await weather_tool.execute(city="London")
"""

from __future__ import annotations

__all__ = [
    "BaseTool",
    "CalculatorTool",
    "ToolExecutionError",
    "ToolExecutionResult",
    "ToolParser",
    "ToolRegistry",
    "WeatherTool",
]

from brok.tools.base import BaseTool, ToolExecutionError, ToolExecutionResult
from brok.tools.calculator import CalculatorTool
from brok.tools.parser import ToolParser
from brok.tools.registry import ToolRegistry
from brok.tools.weather import WeatherTool
