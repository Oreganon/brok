"""Tests for the tool system."""

from __future__ import annotations

import re
from typing import ClassVar

import pytest

from brok.tools import (
    BaseTool,
    CalculatorTool,
    DateTimeTool,
    ToolExecutionError,
    ToolExecutionResult,
    ToolParser,
    ToolRegistry,
    WeatherTool,
)


class TestBaseTool:
    """Test the base tool class."""

    def test_tool_metadata_validation(self):
        """Test that tool metadata is validated on subclass creation."""

        # Should work with valid metadata
        class ValidTool(BaseTool):
            name: ClassVar[str] = "test"
            description: ClassVar[str] = "Test tool"
            parameters: ClassVar[dict] = {"type": "object"}

            async def execute(self, **_kwargs):
                return ToolExecutionResult(success=True, data="test")

        tool = ValidTool()
        assert tool.name == "test"

        # Should fail without name
        with pytest.raises(ValueError, match="must define a 'name'"):

            class InvalidTool1(BaseTool):
                description = "Test tool"
                parameters = {"type": "object"}

                async def execute(self, **kwargs):
                    return ToolExecutionResult(success=True, data="test")

        # Should fail without description
        with pytest.raises(ValueError, match="must define a 'description'"):

            class InvalidTool2(BaseTool):
                name = "test"
                parameters = {"type": "object"}

                async def execute(self, **kwargs):
                    return ToolExecutionResult(success=True, data="test")

    def test_parameter_validation(self):
        """Test parameter validation."""

        class TestTool(BaseTool):
            name = "test"
            description = "Test tool"
            parameters = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "count": {"type": "number"},
                    "active": {"type": "boolean"},
                },
                "required": ["name"],
            }

            async def execute(self, **kwargs):
                return ToolExecutionResult(success=True, data="test")

        tool = TestTool()

        # Valid parameters should pass
        tool.validate_parameters({"name": "test", "count": 5, "active": True})

        # Missing required parameter should fail
        with pytest.raises(ToolExecutionError, match="Missing required parameter"):
            tool.validate_parameters({"count": 5})

        # Wrong type should fail
        with pytest.raises(ToolExecutionError, match="must be a string"):
            tool.validate_parameters({"name": 123})


class TestToolRegistry:
    """Test the tool registry."""

    def test_tool_registration(self):
        """Test registering and accessing tools."""
        registry = ToolRegistry()

        # Create a test tool
        class TestTool(BaseTool):
            name = "test"
            description = "Test tool"
            parameters = {"type": "object"}

            async def execute(self, **kwargs):
                return ToolExecutionResult(success=True, data="test result")

        tool = TestTool()

        # Register tool
        registry.register_tool(tool)
        assert registry.has_tool("test")
        assert "test" in registry.get_available_tools()

        # Get tool
        retrieved_tool = registry.get_tool("test")
        assert retrieved_tool == tool

        # Duplicate registration should fail
        with pytest.raises(ValueError, match="already registered"):
            registry.register_tool(tool)

    def test_tool_unregistration(self):
        """Test unregistering tools."""
        registry = ToolRegistry()

        class TestTool(BaseTool):
            name = "test"
            description = "Test tool"
            parameters = {"type": "object"}

            async def execute(self, **kwargs):
                return ToolExecutionResult(success=True, data="test result")

        tool = TestTool()
        registry.register_tool(tool)

        # Unregister tool
        registry.unregister_tool("test")
        assert not registry.has_tool("test")

        # Unregistering non-existent tool should fail
        with pytest.raises(KeyError, match="not registered"):
            registry.unregister_tool("nonexistent")

    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test executing tools through registry."""
        registry = ToolRegistry()

        class TestTool(BaseTool):
            name = "test"
            description = "Test tool"
            parameters = {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            }

            async def execute(self, **kwargs):
                message = kwargs.get("message", "")
                return ToolExecutionResult(success=True, data=f"Hello {message}")

        tool = TestTool()
        registry.register_tool(tool)

        # Execute tool
        result = await registry.execute_tool("test", {"message": "world"})
        assert result == "Hello world"

        # Invalid parameters should fail
        with pytest.raises(ToolExecutionError):
            await registry.execute_tool("test", {})  # Missing required parameter


class TestToolParser:
    """Test the tool parser."""

    def test_json_format_parsing(self):
        """Test parsing JSON format tool calls."""
        parser = ToolParser(available_tools=["weather", "calculator"])

        # Valid JSON format
        response = '{"tool": "weather", "params": {"city": "London"}}'
        call = parser.parse_response(response)

        assert call is not None
        assert call.tool_name == "weather"
        assert call.parameters == {"city": "London"}

        # Alternative params key
        response = '{"tool": "calculator", "parameters": {"expression": "2+2"}}'
        call = parser.parse_response(response)

        assert call is not None
        assert call.tool_name == "calculator"
        assert call.parameters == {"expression": "2+2"}

    def test_natural_language_parsing(self):
        """Test parsing natural language tool calls."""
        parser = ToolParser(available_tools=["weather", "calculator"])

        # Weather queries
        test_cases = [
            ("What's the weather in London?", "weather", {"city": "London"}),
            ("Check the weather in New York", "weather", {"city": "New York"}),
            ("weather in Tokyo", "weather", {"city": "Tokyo"}),
        ]

        for response, expected_tool, expected_params in test_cases:
            call = parser.parse_response(response)
            assert call is not None
            assert call.tool_name == expected_tool
            assert call.parameters == expected_params

        # Calculator queries
        calc_cases = [
            ("Calculate 2 + 3", "calculator", {"expression": "2 + 3"}),
            (
                "What's 10 * 5?",
                "calculator",
                {"expression": "10 * 5?"},
            ),  # Parser captures the ?
        ]

        for response, expected_tool, expected_params in calc_cases:
            call = parser.parse_response(response)
            assert call is not None
            assert call.tool_name == expected_tool
            assert call.parameters == expected_params

    def test_no_tool_call(self):
        """Test responses with no tool calls."""
        parser = ToolParser(available_tools=["weather", "calculator"])

        responses = [
            "Hello there!",
            "How are you?",
            "I'm doing well, thanks for asking.",
            "That's interesting.",
        ]

        for response in responses:
            call = parser.parse_response(response)
            assert call is None

    def test_exact_json_bug_case(self):
        """Test the exact JSON format that was causing the bug reported by the user."""
        parser = ToolParser(available_tools=["weather", "calculator"])

        # This is the exact response that was getting through without being parsed
        problematic_response = '{"tool": "weather", "params": {"city": "Tokyo"}}'

        call = parser.parse_response(problematic_response)
        assert call is not None
        assert call.tool_name == "weather"
        assert call.parameters == {"city": "Tokyo"}

        # Test variations that might also cause issues
        variations = [
            '{"tool": "weather", "parameters": {"city": "Tokyo"}}',  # parameters vs params
            '{ "tool": "weather", "params": {"city": "Tokyo"} }',  # extra spaces
            '{"tool":"weather","params":{"city":"Tokyo"}}',  # no spaces
            'Let me check: {"tool": "weather", "params": {"city": "Tokyo"}}',  # embedded in text
        ]

        for variation in variations:
            call = parser.parse_response(variation)
            assert call is not None, f"Failed to parse: {variation}"
            assert call.tool_name == "weather"
            assert call.parameters == {"city": "Tokyo"}


class TestWeatherTool:
    """Test the weather tool."""

    @pytest.mark.asyncio
    async def test_weather_validation(self):
        """Test weather tool parameter validation."""
        tool = WeatherTool()

        # Empty city should fail
        result = await tool.execute(city="")
        assert not result.success
        assert "required" in result.error.lower()

        # Missing city should fail
        result = await tool.execute()
        assert not result.success
        assert "required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_weather_tool_format(self):
        """Test weather tool basic functionality."""
        tool = WeatherTool()

        # We can't test actual API calls in unit tests, but we can test the structure
        assert tool.name == "weather"
        assert tool.description
        assert "city" in tool.parameters["properties"]
        assert "city" in tool.parameters["required"]


class TestCalculatorTool:
    """Test the calculator tool."""

    @pytest.mark.asyncio
    async def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        tool = CalculatorTool()

        test_cases = [
            ("2 + 3", 5),
            ("10 - 4", 6),
            ("3 * 4", 12),
            ("15 / 3", 5),
            ("2 ** 3", 8),
        ]

        for expression, expected in test_cases:
            result = await tool.execute(expression=expression)
            assert result.success
            assert f"= {expected}" in result.data

    @pytest.mark.asyncio
    async def test_math_functions(self):
        """Test mathematical functions."""
        tool = CalculatorTool()

        # Test sqrt
        result = await tool.execute(expression="sqrt(16)")
        assert result.success
        assert "= 4" in result.data

        # Test trigonometry
        result = await tool.execute(expression="sin(0)")
        assert result.success
        assert "= 0" in result.data

    @pytest.mark.asyncio
    async def test_calculator_validation(self):
        """Test calculator tool parameter validation."""
        tool = CalculatorTool()

        # Empty expression should fail
        result = await tool.execute(expression="")
        assert not result.success
        assert "required" in result.error.lower()

        # Invalid expression should fail
        result = await tool.execute(expression="invalid expression")
        assert not result.success

        # Division by zero should fail
        result = await tool.execute(expression="1/0")
        assert not result.success
        assert "zero" in result.error.lower()

    @pytest.mark.asyncio
    async def test_expression_cleaning(self):
        """Test expression cleaning and normalization."""
        tool = CalculatorTool()

        # Test text removal
        result = await tool.execute(expression="what is 2 + 3")
        assert result.success
        assert "= 5" in result.data

        # Test implicit multiplication
        result = await tool.execute(expression="2pi")
        assert result.success
        # Should be approximately 6.28 (2 * pi)


class TestDateTimeTool:
    """Test the datetime tool."""

    @pytest.mark.asyncio
    async def test_datetime_tool_metadata(self):
        """Test datetime tool metadata."""
        tool = DateTimeTool()
        assert tool.name == "datetime"
        assert "date and time" in tool.description.lower()
        assert "format" in tool.parameters["properties"]
        assert "timezone" in tool.parameters["properties"]

    @pytest.mark.asyncio
    async def test_readable_format(self):
        """Test readable datetime format (default)."""
        tool = DateTimeTool()
        result = await tool.execute()
        assert result.success
        assert "Current date and time:" in result.data
        assert result.metadata["format_requested"] == "readable"
        assert "timestamp" in result.metadata
        assert "iso_format" in result.metadata

    @pytest.mark.asyncio
    async def test_iso_format(self):
        """Test ISO 8601 datetime format."""
        tool = DateTimeTool()
        result = await tool.execute(format="iso")
        assert result.success
        assert "Current time (ISO 8601):" in result.data
        assert result.metadata["format_requested"] == "iso"
        # Should contain T separator for ISO format
        assert "T" in result.data

    @pytest.mark.asyncio
    async def test_date_only_format(self):
        """Test date-only format."""
        tool = DateTimeTool()
        result = await tool.execute(format="date")
        assert result.success
        assert "Current date:" in result.data
        assert result.metadata["format_requested"] == "date"
        # Should match YYYY-MM-DD pattern
        assert re.search(r"\d{4}-\d{2}-\d{2}", result.data)

    @pytest.mark.asyncio
    async def test_time_only_format(self):
        """Test time-only format."""
        tool = DateTimeTool()
        result = await tool.execute(format="time")
        assert result.success
        assert "Current time:" in result.data
        assert result.metadata["format_requested"] == "time"
        # Should match HH:MM:SS pattern
        assert re.search(r"\d{2}:\d{2}:\d{2}", result.data)

    @pytest.mark.asyncio
    async def test_timestamp_format(self):
        """Test Unix timestamp format."""
        tool = DateTimeTool()
        result = await tool.execute(format="timestamp")
        assert result.success
        assert "Current Unix timestamp:" in result.data
        assert result.metadata["format_requested"] == "timestamp"
        # Should contain digits for timestamp
        assert re.search(r"\d+", result.data)

    @pytest.mark.asyncio
    async def test_invalid_format(self):
        """Test that invalid format falls back to readable."""
        tool = DateTimeTool()
        result = await tool.execute(format="invalid")
        assert result.success
        assert "Current date and time:" in result.data
        assert result.metadata["format_requested"] == "invalid"

    @pytest.mark.asyncio
    async def test_timezone_handling(self):
        """Test timezone parameter handling."""
        tool = DateTimeTool()

        # Test with valid timezone (fallback gracefully if not available)
        result = await tool.execute(timezone="UTC")
        assert result.success
        # Should either include UTC or gracefully fallback to system
        assert result.metadata["timezone"] in ("UTC", "system")

        # Test with invalid timezone (should fallback to system)
        result = await tool.execute(timezone="Invalid/Timezone")
        assert result.success
        assert result.metadata["timezone"] == "system"
