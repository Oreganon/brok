"""Tests for prompt management functionality."""

from datetime import datetime
import logging
from typing import Any, ClassVar
from unittest.mock import patch

import pytest

from brok.prompts import (
    DEFAULT_TEMPLATE,
    PromptTemplate,
    SYSTEM_PROMPT,
    create_custom_template,
    get_prompt_template,
)
from brok.tools.base import BaseTool, ToolExecutionResult
from brok.tools.registry import ToolRegistry


class TestPromptTemplate:
    """Test cases for PromptTemplate class."""

    def test_init_default_values(self):
        """Test PromptTemplate initialization with default values."""
        template = PromptTemplate(system_prompt="Test system")

        assert template.system_prompt == "Test system"
        assert template.user_prefix == "User"
        assert template.assistant_prefix == "Assistant"

    def test_init_custom_values(self):
        """Test PromptTemplate initialization with custom values."""
        template = PromptTemplate(
            system_prompt="Custom system",
            user_prefix="Human",
            assistant_prefix="AI",
        )

        assert template.system_prompt == "Custom system"
        assert template.user_prefix == "Human"
        assert template.assistant_prefix == "AI"

    def test_build_prompt_basic(self):
        """Test building a basic prompt."""
        template = PromptTemplate(system_prompt="You are helpful.")

        result = template.build_prompt("Hello")

        assert "## Instructions" in result
        assert "You are helpful." in result
        assert "## Request" in result
        assert "Hello" in result
        assert "**Assistant:**" in result

    def test_build_prompt_with_context(self):
        """Test building prompt with string context."""
        template = PromptTemplate(system_prompt="You are helpful.")

        result = template.build_prompt("Hello", "Previous chat messages")

        assert "## Instructions" in result
        assert "## Recent Context" in result
        assert "Previous chat messages" in result
        assert "## Request" in result
        assert "Hello" in result

    def test_build_prompt_with_tools_description(self):
        """Test building prompt with legacy tools description."""
        template = PromptTemplate(system_prompt="System")

        result = template.build_prompt("Hello", tools_description="weather: Check weather")

        assert "## Tools Available" in result
        assert "weather: Check weather" in result
        assert "**Usage:**" in result
        assert "Format as JSON" in result

    def test_build_prompt_with_tool_schemas(self):
        """Test building prompt with structured tool schemas."""
        template = PromptTemplate(system_prompt="System")
        tool_schemas = [
            {
                "name": "weather",
                "description": "Get weather info",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"],
                },
            }
        ]

        result = template.build_prompt("Hello", tool_schemas=tool_schemas)

        assert "## Tools Available" in result
        assert "- **weather**: Get weather info" in result
        assert "`city` (string) (required): City name" in result
        assert "**Usage Examples:**" in result

    def test_build_prompt_minimal(self):
        """Test building prompt with minimal content."""
        template = PromptTemplate(system_prompt="")

        result = template.build_prompt("Hello")

        assert "## Request" in result
        assert "Hello" in result
        assert "**Assistant:**" in result
        assert "## Instructions" not in result

    def test_build_prompt_custom_prefixes(self):
        """Test building prompt with custom user/assistant prefixes."""
        template = PromptTemplate(
            system_prompt="System",
            user_prefix="Human",
            assistant_prefix="Bot",
        )

        result = template.build_prompt("Hello")

        assert "**Bot:**" in result


class TestDefaultTemplate:
    """Test cases for default template."""

    def test_default_template_properties(self):
        """Test DEFAULT_TEMPLATE properties."""
        assert DEFAULT_TEMPLATE.system_prompt == SYSTEM_PROMPT
        assert DEFAULT_TEMPLATE.user_prefix == "User"
        assert DEFAULT_TEMPLATE.assistant_prefix == "Assistant"

    def test_system_prompt_content(self):
        """Test that system prompt contains expected content."""
        assert "Brok" in SYSTEM_PROMPT
        assert "1-2 sentences" in SYSTEM_PROMPT
        assert "concisely" in SYSTEM_PROMPT
        assert "Tool Usage:" in SYSTEM_PROMPT


class TestPromptTemplateFactory:
    """Test cases for prompt template factory functions."""

    def test_get_prompt_template(self):
        """Test getting default template."""
        template = get_prompt_template()

        assert template.system_prompt == SYSTEM_PROMPT
        assert template.user_prefix == "User"
        assert template.assistant_prefix == "Assistant"

    def test_create_custom_template(self):
        """Test creating custom template."""
        custom_prompt = "You are a special assistant."
        template = create_custom_template(custom_prompt)

        assert template.system_prompt == custom_prompt
        assert template.user_prefix == "User"
        assert template.assistant_prefix == "Assistant"

    def test_create_custom_template_empty(self):
        """Test creating custom template with empty prompt."""
        template = create_custom_template("")

        assert template.system_prompt == ""
        assert template.user_prefix == "User"
        assert template.assistant_prefix == "Assistant"


class TestMarkdownStructure:
    """Test cases for markdown structure and formatting."""

    def test_markdown_sections_order(self):
        """Test that markdown sections appear in correct order."""
        template = PromptTemplate(system_prompt="Test system")
        tool_schemas = [{"name": "test", "description": "Test tool"}]

        result = template.build_prompt(
            "Hello",
            context="Previous message",
            tool_schemas=tool_schemas
        )

        # Check section order
        instructions_pos = result.find("## Instructions")
        tools_pos = result.find("## Tools Available")
        context_pos = result.find("## Recent Context")
        request_pos = result.find("## Request")

        assert instructions_pos < tools_pos
        assert tools_pos < context_pos
        assert context_pos < request_pos

    def test_tool_schema_formatting(self):
        """Test detailed tool schema formatting."""
        template = PromptTemplate(system_prompt="System")
        tool_schemas = [
            {
                "name": "calculator",
                "description": "Perform math calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression to evaluate"
                        },
                        "precision": {
                            "type": "number",
                            "description": "Number of decimal places"
                        }
                    },
                    "required": ["expression"],
                },
            }
        ]

        result = template.build_prompt("Test", tool_schemas=tool_schemas)

        assert "- **calculator**: Perform math calculations" in result
        assert "`expression` (string) (required): Math expression" in result
        assert "`precision` (number): Number of decimal places" in result
        assert "**Usage Examples:**" in result
        assert '{"tool": "calculator"' in result


class TestPromptLogging:
    """Test cases for prompt logging functionality."""

    def test_prompt_logging_disabled_by_default(self):
        """Test that token logging is disabled by default."""
        template = PromptTemplate(system_prompt="Test system")

        with patch("brok.prompts.logger") as mock_logger:
            result = template.build_prompt("Hello", log_tokens=False)

            mock_logger.log.assert_not_called()
            assert "Hello" in result

    def test_prompt_logging_enabled(self):
        """Test that token logging works when enabled."""
        template = PromptTemplate(system_prompt="Test system")

        with patch("brok.prompts.logger") as mock_logger:
            result = template.build_prompt("Hello", log_tokens=True)

            mock_logger.log.assert_called()
            mock_logger.debug.assert_called()
            assert "Hello" in result

    def test_log_prompt_metrics_performance_levels(self):
        """Test that logging uses appropriate levels based on performance."""
        template = PromptTemplate(system_prompt="Test")

        with (
            patch("brok.prompts.time.perf_counter") as mock_time,
            patch("brok.prompts.logger") as mock_logger,
        ):
            # Mock slow generation (>10ms)
            mock_time.side_effect = [0.0, 0.015]  # 15ms

            template._log_prompt_metrics("test prompt", 15.0, "test input")

            # Should use WARNING level for slow generation
            warning_calls = [
                call
                for call in mock_logger.log.call_args_list
                if call[0][0] == logging.WARNING
            ]
            assert len(warning_calls) > 0

            # Check that performance status is included
            log_message = str(warning_calls[0])
            assert "perf=SLOW" in log_message


class TestPromptIntegration:
    """Integration tests for prompt functionality."""

    def test_llm_provider_integration(self):
        """Test integration with LLM provider-like interface."""
        # Create a simple test tool
        class TestTool(BaseTool):
            name: ClassVar[str] = "test_tool"
            description: ClassVar[str] = "A test tool"
            parameters: ClassVar[dict[str, Any]] = {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Test input parameter",
                    }
                },
                "required": ["input"],
            }

            async def execute(self, **_kwargs: Any) -> ToolExecutionResult:
                return ToolExecutionResult(success=True, data="Test result")

        # Set up tool registry
        registry = ToolRegistry()
        registry.register_tool(TestTool())

        # Test getting tool schemas
        tool_schemas = registry.get_tools_schema()
        
        # Test prompt building with structured tools
        template = get_prompt_template()
        prompt = template.build_prompt("Test input", tool_schemas=tool_schemas)

        # Verify markdown structure includes structured tools
        assert "## Tools Available" in prompt
        assert "- **test_tool**: A test tool" in prompt
        assert "`input` (string) (required): Test input parameter" in prompt

    def test_different_context_formats(self):
        """Test handling of different context formats."""
        template = get_prompt_template()

        # String context
        string_result = template.build_prompt("Hello", context="Previous: Hi there")
        assert "## Recent Context" in string_result
        assert "Previous: Hi there" in string_result

        # No context
        no_context_result = template.build_prompt("Hello")
        assert "## Recent Context" not in no_context_result

    def test_markdown_vs_text_efficiency(self):
        """Test that markdown formatting is reasonably efficient."""
        template = PromptTemplate(system_prompt="Brief system prompt")
        
        # Generate a reasonably complex prompt
        tool_schemas = [
            {
                "name": "weather",
                "description": "Get weather",
                "parameters": {
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]

        result = template.build_prompt(
            "What's the weather?",
            context="Previous conversation",
            tool_schemas=tool_schemas
        )

        # Should be structured but not excessively verbose
        assert len(result) < 1000  # Reasonable upper bound
        assert "##" in result  # Has structure
        assert "**" in result  # Has formatting
