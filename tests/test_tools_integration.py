"""Integration tests for tools with caching functionality."""

from __future__ import annotations

import asyncio

import pytest

from brok.tools import (
    CalculatorTool,
    DateTimeTool,
    InMemoryCache,
    ToolExecutionResult,
    ToolRegistry,
)


class TestToolCachingIntegration:
    """Test cases for tool caching integration."""

    @pytest.fixture
    def shared_cache(self) -> InMemoryCache[ToolExecutionResult]:
        """Create a shared cache for testing."""
        return InMemoryCache[ToolExecutionResult](max_size=50)

    @pytest.fixture
    def registry_with_cache(
        self, shared_cache: InMemoryCache[ToolExecutionResult]
    ) -> ToolRegistry:
        """Create a registry with shared cache."""
        return ToolRegistry(shared_cache=shared_cache)

    @pytest.mark.asyncio
    async def test_calculator_caching(
        self, shared_cache: InMemoryCache[ToolExecutionResult]
    ) -> None:
        """Test that calculator results are cached properly."""
        calc_tool = CalculatorTool()
        calc_tool.set_cache(shared_cache, ttl_seconds=60)

        # First execution - should calculate
        result1 = await calc_tool.execute_cached(expression="2 + 3 * 4")
        assert result1.success
        assert "2 + 3 * 4 = 14" in result1.data
        assert result1.metadata["cache_hit"] is False

        # Second execution - should come from cache
        result2 = await calc_tool.execute_cached(expression="2 + 3 * 4")
        assert result2.success
        assert "2 + 3 * 4 = 14" in result2.data
        assert result2.metadata["cache_hit"] is True

        # Different expression - should calculate again
        result3 = await calc_tool.execute_cached(expression="5 * 6")
        assert result3.success
        assert "5 * 6 = 30" in result3.data
        assert result3.metadata["cache_hit"] is False

    @pytest.mark.asyncio
    async def test_datetime_caching(
        self, shared_cache: InMemoryCache[ToolExecutionResult]
    ) -> None:
        """Test that datetime results are cached properly."""
        datetime_tool = DateTimeTool()
        datetime_tool.set_cache(shared_cache, ttl_seconds=5)  # Short TTL for testing

        # First execution - should fetch time
        result1 = await datetime_tool.execute_cached(format="iso")
        assert result1.success
        assert "Current time (ISO 8601):" in result1.data
        assert result1.metadata["cache_hit"] is False

        # Second execution within TTL - should come from cache
        result2 = await datetime_tool.execute_cached(format="iso")
        assert result2.success
        assert result2.data == result1.data  # Exact same timestamp
        assert result2.metadata["cache_hit"] is True

        # Different format - should be calculated fresh
        result3 = await datetime_tool.execute_cached(format="readable")
        assert result3.success
        assert "Current date and time:" in result3.data
        assert result3.metadata["cache_hit"] is False

    @pytest.mark.asyncio
    async def test_registry_cache_management(
        self, registry_with_cache: ToolRegistry
    ) -> None:
        """Test registry cache management functionality."""
        # Register tools
        calc_tool = CalculatorTool()
        datetime_tool = DateTimeTool()

        registry_with_cache.register_tool(calc_tool)
        registry_with_cache.register_tool(datetime_tool)

        # Execute some operations to populate cache
        await registry_with_cache.execute_tool("calculator", {"expression": "10 + 5"})
        await registry_with_cache.execute_tool("datetime", {"format": "timestamp"})

        # Check cache stats
        stats = registry_with_cache.get_cache_stats()
        assert len(stats) >= 1  # At least one tool should have cache stats

        # Verify cache contains entries
        cache_size = await registry_with_cache._shared_cache.size()
        assert cache_size >= 1

    @pytest.mark.asyncio
    async def test_registry_shared_cache_application(self) -> None:
        """Test that shared cache is applied to registered tools."""
        cache = InMemoryCache[ToolExecutionResult](max_size=10)
        registry = ToolRegistry()

        # Register a tool without cache
        calc_tool = CalculatorTool()
        registry.register_tool(calc_tool)

        # Set shared cache - should apply to existing tools
        registry.set_shared_cache(cache, apply_to_existing=True)

        # Tool should now use shared cache
        result1 = await calc_tool.execute_cached(expression="7 + 8")
        assert result1.metadata["cache_hit"] is False

        result2 = await calc_tool.execute_cached(expression="7 + 8")
        assert result2.metadata["cache_hit"] is True

    @pytest.mark.asyncio
    async def test_complex_calculation_examples(self) -> None:
        """Test complex calculations with caching."""
        cache = InMemoryCache[ToolExecutionResult](max_size=20)
        calc_tool = CalculatorTool()
        calc_tool.set_cache(cache, ttl_seconds=300)

        # Mathematical expressions with different complexity levels
        test_cases = [
            ("2 + 3", "2 + 3 = 5"),
            ("sqrt(16)", "sqrt(16) = 4"),
            ("sin(pi/2)", "sin(pi/2) = 1"),
            ("log(e)", "log(e) = 1"),
            ("factorial(5)", "factorial(5) = 120"),
            ("2**10", "2**10 = 1024"),
            ("ceil(4.2)", "ceil(4.2) = 5"),
            ("min([1, 2, 3])", "min([1, 2, 3]) = 1"),
        ]

        # Execute each calculation twice to test caching
        for expression, expected in test_cases:
            # First execution - fresh calculation
            result1 = await calc_tool.execute_cached(expression=expression)
            assert result1.success
            assert expected in result1.data
            assert result1.metadata["cache_hit"] is False

            # Second execution - from cache
            result2 = await calc_tool.execute_cached(expression=expression)
            assert result2.success
            assert expected in result2.data
            assert result2.metadata["cache_hit"] is True

        # Verify cache has the expected number of entries
        cache_size = await cache.size()
        assert cache_size == len(test_cases)

    @pytest.mark.asyncio
    async def test_datetime_timezone_examples(self) -> None:
        """Test datetime tool with various timezone examples."""
        cache = InMemoryCache[ToolExecutionResult](max_size=15)
        datetime_tool = DateTimeTool()
        datetime_tool.set_cache(cache, ttl_seconds=60)

        # Different timezone and format combinations
        test_cases = [
            {"format": "iso", "timezone": "UTC"},
            {"format": "readable", "timezone": "America/New_York"},
            {"format": "time", "timezone": "Europe/London"},
            {"format": "date", "timezone": "Asia/Tokyo"},
            {"format": "timestamp"},  # No timezone
        ]

        for params in test_cases:
            # First execution - fresh
            result1 = await datetime_tool.execute_cached(**params)
            assert result1.success
            assert result1.metadata["cache_hit"] is False

            # Second execution - cached
            result2 = await datetime_tool.execute_cached(**params)
            assert result2.success
            assert result2.data == result1.data  # Same cached result
            assert result2.metadata["cache_hit"] is True

    @pytest.mark.asyncio
    async def test_concurrent_cache_access(self) -> None:
        """Test concurrent access to cached tools."""
        cache = InMemoryCache[ToolExecutionResult](max_size=10)
        calc_tool = CalculatorTool()
        calc_tool.set_cache(cache, ttl_seconds=60)

        # Simulate concurrent access to the same calculation
        async def calculate(expression: str) -> ToolExecutionResult:
            return await calc_tool.execute_cached(expression=expression)

        # Run 10 concurrent calculations of the same expression
        tasks = [calculate("100 * 200") for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All should return the same result
        first_result = results[0]
        for result in results[1:]:
            assert result.data == first_result.data

        # At least one should be a cache hit (after the first calculation)
        cache_hits = sum(1 for r in results if r.metadata.get("cache_hit", False))
        assert cache_hits >= 1

    @pytest.mark.asyncio
    async def test_cache_key_generation(self) -> None:
        """Test that cache keys are generated consistently."""
        calc_tool = CalculatorTool()

        # Same parameters should generate same key
        key1 = calc_tool._generate_cache_key({"expression": "2 + 3"})
        key2 = calc_tool._generate_cache_key({"expression": "2 + 3"})
        assert key1 == key2

        # Different parameters should generate different keys
        key3 = calc_tool._generate_cache_key({"expression": "3 + 4"})
        assert key1 != key3

        # Parameter order shouldn't matter (for multi-param tools)
        datetime_tool = DateTimeTool()
        key4 = datetime_tool._generate_cache_key({"format": "iso", "timezone": "UTC"})
        key5 = datetime_tool._generate_cache_key({"timezone": "UTC", "format": "iso"})
        assert key4 == key5

    @pytest.mark.asyncio
    async def test_cache_stats_collection(self) -> None:
        """Test cache statistics collection."""
        cache = InMemoryCache[ToolExecutionResult](max_size=5)
        registry = ToolRegistry(shared_cache=cache)

        calc_tool = CalculatorTool()
        datetime_tool = DateTimeTool()

        registry.register_tool(calc_tool)
        registry.register_tool(datetime_tool)

        # Execute several operations
        await calc_tool.execute_cached(expression="1 + 1")
        await calc_tool.execute_cached(expression="1 + 1")  # Cache hit
        await datetime_tool.execute_cached(format="iso")

        # Check individual tool stats
        calc_stats = calc_tool.get_cache_stats()
        assert calc_stats is not None
        assert calc_stats["size"] >= 1

        # Check registry stats
        registry_stats = registry.get_cache_stats()
        assert "calculator" in registry_stats
        assert "datetime" in registry_stats

    @pytest.mark.asyncio
    async def test_error_handling_with_cache(self) -> None:
        """Test that errors are not cached."""
        cache = InMemoryCache[ToolExecutionResult](max_size=5)
        calc_tool = CalculatorTool()
        calc_tool.set_cache(cache, ttl_seconds=60)

        # Execute invalid expression - should fail
        result1 = await calc_tool.execute_cached(expression="invalid syntax")
        assert not result1.success

        # Cache should be empty (errors not cached)
        cache_size = await cache.size()
        assert cache_size == 0

        # Execute valid expression - should succeed and be cached
        result2 = await calc_tool.execute_cached(expression="2 + 2")
        assert result2.success
        assert result2.metadata["cache_hit"] is False

        cache_size = await cache.size()
        assert cache_size == 1


class TestToolExamples:
    """Examples demonstrating tool usage patterns."""

    @pytest.mark.asyncio
    async def test_calculator_comprehensive_examples(self) -> None:
        """Comprehensive examples of calculator tool usage."""
        calc = CalculatorTool()

        # Basic arithmetic
        result = await calc.execute(expression="15 + 25")
        assert result.success and "40" in result.data

        # Order of operations
        result = await calc.execute(expression="2 + 3 * 4")
        assert result.success and "14" in result.data

        # Parentheses
        result = await calc.execute(expression="(2 + 3) * 4")
        assert result.success and "20" in result.data

        # Scientific functions
        result = await calc.execute(expression="sqrt(144)")
        assert result.success and "12" in result.data

        # Trigonometry
        result = await calc.execute(expression="cos(0)")
        assert result.success and "1" in result.data

        # Constants
        result = await calc.execute(expression="pi * 2")
        assert result.success and "6.28" in result.data

        # Complex expressions
        result = await calc.execute(expression="log(exp(5))")
        assert result.success and "5" in result.data

        # Lists and aggregation
        result = await calc.execute(expression="sum([1, 2, 3, 4, 5])")
        assert result.success and "15" in result.data

    @pytest.mark.asyncio
    async def test_datetime_comprehensive_examples(self) -> None:
        """Comprehensive examples of datetime tool usage."""
        dt = DateTimeTool()

        # ISO format
        result = await dt.execute(format="iso")
        assert result.success and "Current time (ISO 8601):" in result.data

        # Human readable
        result = await dt.execute(format="readable")
        assert result.success and "Current date and time:" in result.data

        # Date only
        result = await dt.execute(format="date")
        assert result.success and "Current date:" in result.data

        # Time only
        result = await dt.execute(format="time")
        assert result.success and "Current time:" in result.data

        # Unix timestamp
        result = await dt.execute(format="timestamp")
        assert result.success and "Current Unix timestamp:" in result.data

        # With timezone
        result = await dt.execute(format="iso", timezone="UTC")
        assert result.success and "(UTC)" in result.data

        # Different timezones
        result = await dt.execute(format="readable", timezone="America/New_York")
        assert result.success and "(America/New_York)" in result.data

    @pytest.mark.asyncio
    async def test_registry_usage_examples(self) -> None:
        """Examples of using the tool registry."""
        # Create registry with caching
        cache = InMemoryCache[ToolExecutionResult](max_size=100)
        registry = ToolRegistry(shared_cache=cache)

        # Register tools
        registry.register_tool(CalculatorTool())
        registry.register_tool(DateTimeTool())

        # List available tools
        tools = registry.get_available_tools()
        assert "calculator" in tools
        assert "datetime" in tools

        # Execute tools through registry
        calc_result = await registry.execute_tool(
            "calculator", {"expression": "sum([1, 1, 2, 3, 5, 8])"}
        )
        assert "20" in calc_result

        time_result = await registry.execute_tool(
            "datetime", {"format": "readable", "timezone": "Europe/London"}
        )
        assert "Current date and time:" in time_result

        # Get tool descriptions
        descriptions = registry.get_tools_description()
        assert "calculator" in descriptions
        assert "datetime" in descriptions

        # Check schemas for LLM integration
        schemas = registry.get_tools_schema()
        assert len(schemas) == 2
        assert any(schema["name"] == "calculator" for schema in schemas)
        assert any(schema["name"] == "datetime" for schema in schemas)
