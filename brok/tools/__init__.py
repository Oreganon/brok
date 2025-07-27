"""Tool system for brok chatbot.

This module provides a framework for extending the bot with tools that can
access external APIs and services to provide real-time information.

Available tools:
- Weather: Get current weather and forecasts (requires aiohttp)
- Calculator: Perform mathematical calculations
- DateTime: Get current date and time information
- Cache: Caching system for tool results

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
    "Cache",
    "CacheEntry",
    "CalculatorTool",
    "DateTimeTool",
    "InMemoryCache",
    "ToolExecutionError",
    "ToolExecutionResult",
    "ToolParser",
    "ToolRegistry",
]

from brok.tools.base import BaseTool, ToolExecutionError, ToolExecutionResult
from brok.tools.cache import Cache, CacheEntry, InMemoryCache
from brok.tools.calculator import CalculatorTool
from brok.tools.datetime import DateTimeTool
from brok.tools.parser import ToolParser
from brok.tools.registry import ToolRegistry

# Import WeatherTool conditionally to handle missing aiohttp dependency
try:
    from brok.tools.weather import WeatherTool  # noqa: F401

    __all__.append("WeatherTool")
    _WEATHER_AVAILABLE = True
except ImportError:
    _WEATHER_AVAILABLE = False
