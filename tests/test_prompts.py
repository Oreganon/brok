"""Tests for prompt management functionality."""

from datetime import datetime
import logging
from typing import Any, ClassVar
from unittest.mock import patch
import xml.etree.ElementTree as ET

import pytest

from brok.chat import ContextMessage
from brok.llm.base import LLMConfig
from brok.llm.ollama import OllamaProvider
from brok.prompts import (
    ADAPTIVE_TEMPLATE,
    CONCISE_TEMPLATE,
    DEFAULT_ADAPTIVE_PROMPT,
    DEFAULT_CONCISE_PROMPT,
    DEFAULT_DETAILED_PROMPT,
    DETAILED_TEMPLATE,
    LightweightXMLPromptTemplate,  # New import for KEP-002 Increment D
    PromptTemplate,
    XMLPromptTemplate,
    create_custom_template,
    get_lightweight_xml_template,  # New import
    get_optimal_xml_template,  # New import
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

    def test_build_prompt_with_system_and_context(self):
        """Test building prompt with system prompt and context."""
        template = PromptTemplate(system_prompt="You are helpful.")

        result = template.build_prompt("Hello", "Previous chat messages")

        expected = "System: You are helpful.\n\nContext:\nPrevious chat messages\n\nUser: Hello\n\nAssistant:"
        assert result == expected

    def test_build_prompt_with_system_only(self):
        """Test building prompt with system prompt but no context."""
        template = PromptTemplate(system_prompt="You are helpful.")

        result = template.build_prompt("Hello", None)

        expected = "System: You are helpful.\n\nUser: Hello\n\nAssistant:"
        assert result == expected

    def test_build_prompt_no_system_with_context(self):
        """Test building prompt with context but no system prompt."""
        template = PromptTemplate(system_prompt="")

        result = template.build_prompt("Hello", "Previous chat")

        expected = "Context:\nPrevious chat\n\nUser: Hello\n\nAssistant:"
        assert result == expected

    def test_build_prompt_minimal(self):
        """Test building prompt with minimal content."""
        template = PromptTemplate(system_prompt="")

        result = template.build_prompt("Hello", None)

        expected = "User: Hello\n\nAssistant:"
        assert result == expected

    def test_build_prompt_empty_context_string(self):
        """Test building prompt with empty context string."""
        template = PromptTemplate(system_prompt="System")

        result = template.build_prompt("Hello", "")

        expected = "System: System\n\nUser: Hello\n\nAssistant:"
        assert result == expected

    def test_build_prompt_whitespace_system(self):
        """Test building prompt with whitespace-only system prompt."""
        template = PromptTemplate(system_prompt="   \n\t  ")

        result = template.build_prompt("Hello", None)

        expected = "User: Hello\n\nAssistant:"
        assert result == expected

    def test_build_prompt_custom_prefixes(self):
        """Test building prompt with custom user/assistant prefixes."""
        template = PromptTemplate(
            system_prompt="System",
            user_prefix="Human",
            assistant_prefix="Bot",
        )

        result = template.build_prompt("Hello", None)

        expected = "System: System\n\nHuman: Hello\n\nBot:"
        assert result == expected


class TestDefaultTemplates:
    """Test cases for default template constants."""

    def test_concise_template(self):
        """Test CONCISE_TEMPLATE properties."""
        assert CONCISE_TEMPLATE.system_prompt == DEFAULT_CONCISE_PROMPT
        assert CONCISE_TEMPLATE.user_prefix == "User"
        assert CONCISE_TEMPLATE.assistant_prefix == "Assistant"

    def test_detailed_template(self):
        """Test DETAILED_TEMPLATE properties."""
        assert DETAILED_TEMPLATE.system_prompt == DEFAULT_DETAILED_PROMPT
        assert DETAILED_TEMPLATE.user_prefix == "User"
        assert DETAILED_TEMPLATE.assistant_prefix == "Assistant"

    def test_adaptive_template(self):
        """Test ADAPTIVE_TEMPLATE properties."""
        assert ADAPTIVE_TEMPLATE.system_prompt == DEFAULT_ADAPTIVE_PROMPT
        assert ADAPTIVE_TEMPLATE.user_prefix == "User"
        assert ADAPTIVE_TEMPLATE.assistant_prefix == "Assistant"

    def test_default_prompts_contain_key_instructions(self):
        """Test that default prompts contain expected key instructions."""
        # Concise should encourage brevity
        assert "concise" in DEFAULT_CONCISE_PROMPT.lower()
        assert "2-3 sentences" in DEFAULT_CONCISE_PROMPT
        assert "brief" in DEFAULT_CONCISE_PROMPT.lower()

        # Detailed should encourage thoroughness
        assert "detailed" in DEFAULT_DETAILED_PROMPT.lower()
        assert "thorough" in DEFAULT_DETAILED_PROMPT.lower()
        assert "comprehensive" in DEFAULT_DETAILED_PROMPT.lower()

        # Adaptive should mention adapting response length
        assert "adapt" in DEFAULT_ADAPTIVE_PROMPT.lower()
        assert "complexity" in DEFAULT_ADAPTIVE_PROMPT.lower()
        assert "brief" in DEFAULT_ADAPTIVE_PROMPT.lower()

    def test_all_prompts_avoid_self_reference(self):
        """Test that all default prompts avoid mentioning the bot's own name."""
        prompts = [
            DEFAULT_CONCISE_PROMPT,
            DEFAULT_DETAILED_PROMPT,
            DEFAULT_ADAPTIVE_PROMPT,
        ]

        for prompt in prompts:
            # Prompts should not mention "brok" as per the design change
            assert "brok" not in prompt.lower()
            # But should still be chat room context
            assert "chat room" in prompt.lower()
            # And should include the instruction to not mention own name
            assert "never mention your own name" in prompt.lower()


class TestPromptTemplateFactory:
    """Test cases for prompt template factory functions."""

    def test_get_prompt_template_concise(self):
        """Test getting concise template."""
        template = get_prompt_template("concise")

        assert template.system_prompt == DEFAULT_CONCISE_PROMPT
        assert template.user_prefix == "User"
        assert template.assistant_prefix == "Assistant"

    def test_get_prompt_template_detailed(self):
        """Test getting detailed template."""
        template = get_prompt_template("detailed")

        assert template.system_prompt == DEFAULT_DETAILED_PROMPT
        assert template.user_prefix == "User"
        assert template.assistant_prefix == "Assistant"

    def test_get_prompt_template_adaptive(self):
        """Test getting adaptive template."""
        template = get_prompt_template("adaptive")

        assert template.system_prompt == DEFAULT_ADAPTIVE_PROMPT
        assert template.user_prefix == "User"
        assert template.assistant_prefix == "Assistant"

    def test_get_prompt_template_default(self):
        """Test getting template with default parameter."""
        template = get_prompt_template()

        assert template.system_prompt == DEFAULT_CONCISE_PROMPT

    def test_get_prompt_template_invalid_style(self):
        """Test getting template with invalid style raises ValueError."""
        with pytest.raises(ValueError, match="Unknown prompt style: invalid"):
            get_prompt_template("invalid")

    def test_get_prompt_template_case_sensitivity(self):
        """Test that get_prompt_template is case-sensitive."""
        with pytest.raises(ValueError):
            get_prompt_template("CONCISE")

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


class TestPromptIntegration:
    """Integration tests for prompt functionality."""

    def test_different_templates_produce_different_prompts(self):
        """Test that different template styles produce different prompts."""
        user_input = "What is Python?"
        context = "Programming discussion"

        concise = get_prompt_template("concise")
        detailed = get_prompt_template("detailed")
        adaptive = get_prompt_template("adaptive")

        concise_prompt = concise.build_prompt(user_input, context)
        detailed_prompt = detailed.build_prompt(user_input, context)
        adaptive_prompt = adaptive.build_prompt(user_input, context)

        # All should have different system prompts
        assert concise_prompt != detailed_prompt
        assert detailed_prompt != adaptive_prompt
        assert concise_prompt != adaptive_prompt

        # But all should contain the user input and context
        for prompt in [concise_prompt, detailed_prompt, adaptive_prompt]:
            assert user_input in prompt
            assert context in prompt
            assert "User:" in prompt
            assert "Assistant:" in prompt

    def test_custom_template_integration(self):
        """Test custom template integration."""
        custom_system = "You are a math tutor. Always show step-by-step solutions."
        template = create_custom_template(custom_system)

        prompt = template.build_prompt("Solve 2x + 5 = 11", None)

        assert custom_system in prompt
        assert "Solve 2x + 5 = 11" in prompt
        assert "User:" in prompt
        assert "Assistant:" in prompt

    def test_template_with_multiline_content(self):
        """Test template with multiline user input and context."""
        template = get_prompt_template("concise")
        user_input = "First line\nSecond line\nThird line"
        context = "Previous message 1\nPrevious message 2"

        prompt = template.build_prompt(user_input, context)

        assert user_input in prompt
        assert context in prompt
        assert "System:" in prompt
        assert "Context:" in prompt
        assert "User:" in prompt
        assert "Assistant:" in prompt


class TestXMLPromptTemplate:
    """Test cases for XMLPromptTemplate class (KEP-002)."""

    def test_xml_template_inherits_from_prompt_template(self):
        """Test that XMLPromptTemplate inherits from PromptTemplate."""
        # Act
        xml_template = XMLPromptTemplate(system_prompt="Test system")

        # Assert
        assert isinstance(xml_template, PromptTemplate)
        assert xml_template.system_prompt == "Test system"
        assert xml_template.user_prefix == "User"
        assert xml_template.assistant_prefix == "Assistant"

    def test_xml_formatting_disabled_identical_to_base_template(self):
        """Test that XMLPromptTemplate with xml_formatting=False produces identical output to PromptTemplate."""
        # Arrange
        system_prompt = "You are a helpful assistant."
        user_input = "Hello, how are you?"
        context = "Previous conversation here"
        tools_desc = "Available tools: weather"

        base_template = PromptTemplate(system_prompt=system_prompt)
        xml_template = XMLPromptTemplate(system_prompt=system_prompt)

        # Act
        base_output = base_template.build_prompt(user_input, context, tools_desc)
        xml_output = xml_template.build_prompt(
            user_input, context, tools_desc, xml_formatting=False
        )

        # Assert - This is the critical round-trip test for backward compatibility
        assert xml_output == base_output

    def test_xml_formatting_disabled_no_context_no_tools(self):
        """Test XMLPromptTemplate without context or tools produces identical output when XML disabled."""
        # Arrange
        system_prompt = "You are brok."
        user_input = "What is 2+2?"

        base_template = PromptTemplate(system_prompt=system_prompt)
        xml_template = XMLPromptTemplate(system_prompt=system_prompt)

        # Act
        base_output = base_template.build_prompt(user_input)
        xml_output = xml_template.build_prompt(user_input, xml_formatting=False)

        # Assert
        assert xml_output == base_output

    def test_xml_formatting_enabled_creates_structured_xml(self):
        """Test that XMLPromptTemplate with xml_formatting=True creates structured XML."""
        # Arrange
        xml_template = XMLPromptTemplate(
            system_prompt="You are brok", user_prefix="Human", assistant_prefix="AI"
        )
        user_input = "Hello"

        # Act
        result = xml_template.build_prompt(user_input, xml_formatting=True)

        # Assert
        assert result.startswith('<prompt version="1.0">')
        assert '<instructions role="assistant" name="brok">' in result
        assert "You are brok" in result
        assert "Respond only in plain text" in result
        assert "<user_input>Hello</user_input>" in result
        assert "<response_prompt>AI:</response_prompt>" in result
        assert "</prompt>" in result

    def test_xml_formatting_with_context(self):
        """Test XML formatting includes context section."""
        # Arrange
        xml_template = XMLPromptTemplate(system_prompt="System")
        context = "Previous chat messages"

        # Act
        result = xml_template.build_prompt(
            "Hello", context=context, xml_formatting=True
        )

        # Assert
        assert (
            '<context window_size="10" format="legacy">Previous chat messages</context>'
            in result
        )

    def test_xml_formatting_with_tools(self):
        """Test XML formatting includes tools section with description and usage."""
        # Arrange
        xml_template = XMLPromptTemplate(system_prompt="System")
        tools_desc = "weather tool for checking weather"

        # Act
        result = xml_template.build_prompt(
            "Hello", tools_description=tools_desc, xml_formatting=True
        )

        # Assert
        assert '<tools count="1">' in result
        assert "<description>weather tool for checking weather</description>" in result
        assert "<usage_instructions>" in result
        assert "You can use tools by responding in this format:" in result

    def test_xml_formatting_with_all_sections(self):
        """Test XML formatting with system, tools, context, and request sections."""
        # Arrange
        xml_template = XMLPromptTemplate(
            system_prompt="You are helpful",
            user_prefix="User",
            assistant_prefix="Assistant",
        )
        context = "Chat history here"
        tools_desc = "Available tools"
        user_input = "What's the weather?"

        # Act
        result = xml_template.build_prompt(
            user_input,
            context=context,
            tools_description=tools_desc,
            xml_formatting=True,
        )

        # Assert
        # Check all sections are present
        assert '<instructions role="assistant" name="brok">' in result
        assert "You are helpful" in result
        assert "Respond only in plain text" in result
        assert '<tools count="1">' in result
        assert (
            '<context window_size="10" format="legacy">Chat history here</context>'
            in result
        )
        assert '<request sender="user">' in result
        assert "<user_input>What's the weather?</user_input>" in result
        assert "<response_prompt>Assistant:</response_prompt>" in result

    def test_xml_formatting_handles_empty_system_prompt(self):
        """Test XML formatting handles empty system prompt gracefully."""
        # Arrange
        xml_template = XMLPromptTemplate(system_prompt="   \n  ")

        # Act
        result = xml_template.build_prompt("Hello", xml_formatting=True)

        # Assert
        assert "<system" not in result  # No system section should be created
        assert "<user_input>Hello</user_input>" in result

    def test_xml_formatting_handles_empty_context(self):
        """Test XML formatting handles empty context gracefully."""
        # Arrange
        xml_template = XMLPromptTemplate(system_prompt="System")

        # Act
        result = xml_template.build_prompt(
            "Hello", context="   \n  ", xml_formatting=True
        )

        # Assert
        assert "<context" not in result  # No context section should be created

    def test_xml_formatting_handles_empty_tools(self):
        """Test XML formatting handles empty tools description gracefully."""
        # Arrange
        xml_template = XMLPromptTemplate(system_prompt="System")

        # Act
        result = xml_template.build_prompt(
            "Hello", tools_description="   ", xml_formatting=True
        )

        # Assert
        assert "<tools" not in result  # No tools section should be created

    def test_xml_output_is_well_formed(self):
        """Test that XML output is well-formed and can be parsed."""
        # Arrange
        xml_template = XMLPromptTemplate(system_prompt="System")

        # Act
        result = xml_template.build_prompt("Hello", xml_formatting=True)

        # Assert - This will raise an exception if XML is malformed
        root = ET.fromstring(result)
        assert root.tag == "prompt"
        assert root.get("version") == "1.0"

    @pytest.mark.parametrize(
        "system_prompt,user_input,context,tools_desc",
        [
            ("System 1", "Input 1", None, None),
            ("System 2", "Input 2", "Context", None),
            ("System 3", "Input 3", None, "Tools"),
            ("System 4", "Input 4", "Context", "Tools"),
            ("", "Input 5", "Context", "Tools"),  # Empty system
        ],
    )
    def test_backward_compatibility_parametrized(
        self, system_prompt, user_input, context, tools_desc
    ):
        """Parametrized test ensuring backward compatibility across various input combinations."""
        # Arrange
        base_template = PromptTemplate(system_prompt=system_prompt)
        xml_template = XMLPromptTemplate(system_prompt=system_prompt)

        # Act
        base_output = base_template.build_prompt(user_input, context, tools_desc)
        xml_output = xml_template.build_prompt(
            user_input, context, tools_desc, xml_formatting=False
        )

        # Assert
        assert xml_output == base_output


class TestXMLPromptTemplateStructuredContext:
    """Test cases for XMLPromptTemplate structured context (KEP-002 Increment B)."""

    def test_structured_context_creates_individual_message_elements(self):
        """Test that structured context creates individual <message> elements with metadata."""
        # Arrange
        xml_template = XMLPromptTemplate(system_prompt="System")
        context_messages = [
            ContextMessage(
                content="Hello world",
                sender="alice",
                timestamp=datetime(2025, 1, 26, 15, 30, 0),
                is_bot=False,
                message_id="msg_1",
            ),
            ContextMessage(
                content="Hi there!",
                sender="brok",
                timestamp=datetime(2025, 1, 26, 15, 30, 15),
                is_bot=True,
                message_id="msg_2",
            ),
        ]

        # Act
        result = xml_template.build_prompt(
            "How are you?", xml_formatting=True, context_messages=context_messages
        )

        # Assert
        assert '<context window_size="2" format="structured">' in result
        assert (
            '<message sender="alice" timestamp="2025-01-26T15:30:00" type="user_message" id="msg_1">Hello world</message>'
            in result
        )
        assert (
            '<message sender="brok" timestamp="2025-01-26T15:30:15" type="bot_response" id="msg_2">Hi there!</message>'
            in result
        )

    def test_structured_context_preferred_over_string_context(self):
        """Test that structured context is used when both structured and string context are provided."""
        # Arrange
        xml_template = XMLPromptTemplate(system_prompt="System")
        string_context = "alice: Should not appear"
        context_messages = [
            ContextMessage(
                content="Should appear",
                sender="alice",
                timestamp=datetime.now(),
                is_bot=False,
            )
        ]

        # Act
        result = xml_template.build_prompt(
            "Hello",
            context=string_context,
            xml_formatting=True,
            context_messages=context_messages,
        )

        # Assert
        assert "Should appear" in result
        assert "Should not appear" not in result
        assert 'format="structured"' in result
        assert 'format="legacy"' not in result

    def test_legacy_string_context_fallback(self):
        """Test that legacy string context is used when no structured context is provided."""
        # Arrange
        xml_template = XMLPromptTemplate(system_prompt="System")
        string_context = "alice: Legacy context message"

        # Act
        result = xml_template.build_prompt(
            "Hello", context=string_context, xml_formatting=True
        )

        # Assert
        assert "Legacy context message" in result
        assert 'format="legacy"' in result
        assert 'format="structured"' not in result

    def test_no_context_section_when_both_empty(self):
        """Test that no context section is created when both context types are empty."""
        # Arrange
        xml_template = XMLPromptTemplate(system_prompt="System")

        # Act
        result = xml_template.build_prompt("Hello", xml_formatting=True)

        # Assert
        assert "<context" not in result

    def test_empty_structured_context_creates_no_section(self):
        """Test that empty structured context list creates no context section."""
        # Arrange
        xml_template = XMLPromptTemplate(system_prompt="System")

        # Act
        result = xml_template.build_prompt(
            "Hello", xml_formatting=True, context_messages=[]
        )

        # Assert
        assert "<context" not in result

    def test_structured_context_xml_is_well_formed(self):
        """Test that structured context creates well-formed XML."""
        # Arrange
        xml_template = XMLPromptTemplate(system_prompt="System")
        context_messages = [
            ContextMessage(
                content="Test message with <special> & characters",
                sender="user_with_underscore",
                timestamp=datetime(2025, 1, 26, 15, 30, 0),
                is_bot=False,
            )
        ]

        # Act
        result = xml_template.build_prompt(
            "Hello", xml_formatting=True, context_messages=context_messages
        )

        # Assert - This will raise an exception if XML is malformed
        root = ET.fromstring(result)
        context_elem = root.find("context")
        assert context_elem is not None
        assert context_elem.get("format") == "structured"
        assert context_elem.get("window_size") == "1"

        message_elem = context_elem.find("message")
        assert message_elem is not None
        assert message_elem.get("sender") == "user_with_underscore"
        assert message_elem.get("type") == "user_message"
        assert message_elem.text == "Test message with <special> & characters"

    def test_backward_compatibility_with_xml_formatting_disabled(self):
        """Test that structured context parameters don't affect output when XML formatting is disabled."""
        # Arrange
        system_prompt = "You are helpful"
        user_input = "Hello"
        string_context = "alice: Previous message"
        context_messages = [
            ContextMessage(
                content="Should be ignored",
                sender="alice",
                timestamp=datetime.now(),
                is_bot=False,
            )
        ]

        base_template = PromptTemplate(system_prompt=system_prompt)
        xml_template = XMLPromptTemplate(system_prompt=system_prompt)

        # Act
        base_output = base_template.build_prompt(user_input, string_context)
        xml_output = xml_template.build_prompt(
            user_input,
            context=string_context,
            xml_formatting=False,
            context_messages=context_messages,  # Should be ignored
        )

        # Assert - Output should be identical (backward compatibility)
        assert xml_output == base_output
        assert "Should be ignored" not in xml_output
        assert "Previous message" in xml_output


class TestXMLPromptTemplateStructuredTools:
    """Test cases for XMLPromptTemplate structured tools (KEP-002 Increment C)."""

    def test_structured_tools_creates_individual_tool_elements(self):
        """Test that structured tools create individual <tool> elements with metadata."""
        # Arrange
        xml_template = XMLPromptTemplate(system_prompt="System")
        tool_schemas = [
            {
                "name": "weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name for weather lookup",
                        }
                    },
                    "required": ["city"],
                },
            }
        ]

        # Act
        result = xml_template.build_prompt(
            "What's the weather?", tool_schemas=tool_schemas, xml_formatting=True
        )

        # Assert
        assert '<tools count="1" format="structured">' in result
        assert '<tool name="weather" category="function">' in result
        assert "<description>Get current weather for a city</description>" in result
        assert "<parameters>" in result
        assert '<parameter name="city" type="string" required="true">' in result
        assert "City name" in result  # Updated to match simplified description

    def test_structured_tools_with_multiple_tools(self):
        """Test structured tools with multiple tool definitions."""
        # Arrange
        xml_template = XMLPromptTemplate(system_prompt="System")
        tool_schemas = [
            {
                "name": "weather",
                "description": "Get weather info",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
            {
                "name": "calculator",
                "description": "Perform calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                        "precision": {"type": "number"},
                    },
                    "required": ["expression"],
                },
            },
        ]

        # Act
        result = xml_template.build_prompt(
            "Help me", tool_schemas=tool_schemas, xml_formatting=True
        )

        # Assert
        assert '<tools count="2" format="structured">' in result
        assert '<tool name="weather" category="function">' in result
        assert '<tool name="calculator" category="function">' in result
        assert "Get weather info" in result
        assert "Perform calculations" in result

    def test_structured_tools_includes_usage_examples(self):
        """Test that structured tools include JSON usage examples."""
        # Arrange
        xml_template = XMLPromptTemplate(system_prompt="System")
        tool_schemas = [
            {
                "name": "weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]

        # Act
        result = xml_template.build_prompt(
            "Test", tool_schemas=tool_schemas, xml_formatting=True
        )

        # Assert
        assert "<usage_example>" in result
        assert '{"tool": "weather", "params": {"city": "London"}}' in result

    def test_structured_tools_fallback_to_legacy_format(self):
        """Test that when tools_description is provided, it falls back to legacy format."""
        # Arrange
        xml_template = XMLPromptTemplate(system_prompt="System")

        # Act - Use legacy tools_description parameter
        result = xml_template.build_prompt(
            "Test", tools_description="Available tools: weather", xml_formatting=True
        )

        # Assert - Should use legacy format
        assert '<tools count="1">' in result
        assert "<description>Available tools: weather</description>" in result
        assert "<usage_instructions>" in result
        # Should NOT have structured tool elements
        assert "<tool name=" not in result

    def test_backward_compatibility_with_xml_formatting_disabled(self):
        """Test that structured tools are ignored when XML formatting is disabled."""
        # Arrange
        xml_template = XMLPromptTemplate(system_prompt="System")
        tool_schemas = [{"name": "weather", "description": "Get weather"}]

        # Act
        result = xml_template.build_prompt(
            "Test", tool_schemas=tool_schemas, xml_formatting=False
        )

        # Assert - Should use parent class behavior (no tools section at all)
        assert "<tools>" not in result
        assert "<tool>" not in result

    def test_empty_tool_schemas_creates_no_tools_section(self):
        """Test that empty tool schemas don't create a tools section."""
        # Arrange
        xml_template = XMLPromptTemplate(system_prompt="System")

        # Act
        result = xml_template.build_prompt("Test", tool_schemas=[], xml_formatting=True)

        # Assert
        assert "<tools>" not in result

    def test_structured_tools_xml_is_well_formed(self):
        """Test that generated structured tools XML is well-formed."""
        # Arrange
        xml_template = XMLPromptTemplate(system_prompt="System")
        tool_schemas = [
            {
                "name": "weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]

        # Act
        result = xml_template.build_prompt(
            "Test", tool_schemas=tool_schemas, xml_formatting=True
        )

        # Assert - Should be parseable XML
        try:
            ET.fromstring(result)
        except ET.ParseError as e:
            pytest.fail(f"Generated XML is not well-formed: {e}")


class TestStructuredToolsIntegration:
    """Integration tests for structured tools with LLM providers (KEP-002 Increment C)."""

    def test_llm_provider_get_tools_schema_integration(self):
        """Test integration between ToolRegistry, LLMProvider, and XMLPromptTemplate."""

        # Create a simple test tool
        class TestTool(BaseTool):
            name: ClassVar[str] = "test_tool"
            description: ClassVar[str] = "A test tool for integration testing"
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

        # Create Ollama provider with XML template
        config = LLMConfig(
            model_name="test-model", max_tokens=100, temperature=0.7, timeout_seconds=30
        )
        xml_template = XMLPromptTemplate(system_prompt="Test system")
        provider = OllamaProvider(
            "http://localhost:11434", "test-model", config, xml_template
        )
        provider.set_tool_registry(registry)

        # Test that provider can get structured tools
        tool_schemas = provider.get_tools_schema()

        # Verify structure
        assert len(tool_schemas) == 1
        assert tool_schemas[0]["name"] == "test_tool"
        assert tool_schemas[0]["description"] == "A test tool for integration testing"
        assert "parameters" in tool_schemas[0]

        # Test prompt building with structured tools
        prompt = provider._build_prompt("Test input", None, None)

        # Verify XML structure includes structured tools
        assert '<tools count="1" format="structured">' in prompt
        assert '<tool name="test_tool" category="function">' in prompt
        assert "A test tool for integration testing" in prompt


class TestLightweightXMLPromptTemplate:
    """Test cases for LightweightXMLPromptTemplate class (KEP-002 Increment D)."""

    def test_init_with_token_counter(self):
        """Test LightweightXMLPromptTemplate initialization includes token counter."""
        template = LightweightXMLPromptTemplate(system_prompt="Test system")

        assert hasattr(template, "_token_counter")
        assert template._token_counter is not None

    def test_build_prompt_minimal_xml_structure(self):
        """Test lightweight template produces minimal XML structure."""
        template = LightweightXMLPromptTemplate(system_prompt="Be helpful.")

        result = template.build_prompt(
            user_input="Hello",
            xml_formatting=True,
        )

        # Should be valid XML
        root = ET.fromstring(result)
        assert root.tag == "prompt"

        # Should have minimal structure - no verbose instructions
        system_elem = root.find("system")
        assert system_elem is not None
        assert system_elem.text == "Be helpful."

        # Should not contain verbose XML warnings
        assert "Do not include any XML tags" not in result

    def test_build_prompt_compact_formatting(self):
        """Test lightweight template uses compact formatting."""
        template = LightweightXMLPromptTemplate(system_prompt="System")

        result = template.build_prompt(
            user_input="Test",
            xml_formatting=True,
        )

        # Should have no indentation (compact formatting)
        lines = result.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]

        # Most lines should not start with whitespace (no indentation)
        lines_with_indent = [line for line in non_empty_lines if line.startswith(" ")]
        indent_ratio = len(lines_with_indent) / len(non_empty_lines)

        # Should have minimal indentation for compactness
        assert indent_ratio < 0.5

    def test_compact_tools_minimal_structure(self):
        """Test lightweight template creates minimal tool descriptions."""
        template = LightweightXMLPromptTemplate(system_prompt="System")

        tool_schemas = [
            {
                "name": "weather",
                "description": "Get weather info",
                "parameters": {
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "units": {"type": "string", "description": "Temperature units"},
                    },
                    "required": ["city"],
                },
            }
        ]

        result = template.build_prompt(
            user_input="Test",
            xml_formatting=True,
            tool_schemas=tool_schemas,
        )

        # Should contain tool but in compact format
        assert "weather" in result
        assert "Get weather info" in result

        # Should include compact parameter format
        assert "city*:string" in result  # Required parameter with marker
        assert "units:string" in result  # Optional parameter

        # Should not have verbose usage instructions
        assert "TOOL USAGE GUIDELINES:" not in result
        assert "validation_rules" not in result

    def test_compact_context_ultra_minimal(self):
        """Test lightweight template creates ultra-minimal context."""
        template = LightweightXMLPromptTemplate(system_prompt="System")

        context_messages = [
            ContextMessage("Hello there", "alice", datetime.now(), False),
            ContextMessage("Hi back", "brok", datetime.now(), True),
        ]

        result = template.build_prompt(
            user_input="Test",
            xml_formatting=True,
            context_messages=context_messages,
        )

        # Should use ultra-short attributes
        assert 'u="alice"' in result  # Ultra-short sender attribute
        assert 't="u"' in result  # Ultra-short type (u=user)
        assert 't="b"' in result  # Ultra-short type (b=bot)

        # Should not have verbose attributes like "timestamp" or "type"
        assert "timestamp=" not in result
        assert "type=" not in result

    def test_measure_token_efficiency_basic(self):
        """Test token efficiency measurement functionality."""
        template = LightweightXMLPromptTemplate(system_prompt="Be helpful.")

        analysis = template.measure_token_efficiency(
            user_input="What's the weather?",
        )

        # Should return comprehensive analysis
        assert "xml_tokens" in analysis
        assert "text_tokens" in analysis
        assert "token_overhead_percent" in analysis
        assert "xml_is_efficient" in analysis
        assert "recommended_for_2b" in analysis

        # Values should be reasonable
        assert analysis["xml_tokens"] > 0
        assert analysis["text_tokens"] > 0
        assert isinstance(analysis["recommended_for_2b"], bool)

    def test_measure_token_efficiency_with_context(self):
        """Test token efficiency with context messages."""
        template = LightweightXMLPromptTemplate(system_prompt="Be helpful.")

        context_messages = [
            ContextMessage("Previous message", "user1", datetime.now(), False),
        ]

        analysis = template.measure_token_efficiency(
            user_input="Follow up question",
            context_messages=context_messages,
        )

        # Should account for context in efficiency calculation
        assert analysis["xml_tokens"] > analysis["text_tokens"]
        assert "generation_time_ms" in analysis

    def test_backward_compatibility_when_xml_disabled(self):
        """Test lightweight template maintains compatibility when XML disabled."""
        template = LightweightXMLPromptTemplate(system_prompt="System prompt")

        result = template.build_prompt(
            user_input="Hello",
            xml_formatting=False,  # Disabled
        )

        # Should be identical to regular template when XML disabled
        regular_template = PromptTemplate(system_prompt="System prompt")
        expected = regular_template.build_prompt("Hello")

        assert result == expected

    def test_context_messages_to_text_conversion(self):
        """Test conversion of context messages to text format."""
        template = LightweightXMLPromptTemplate(system_prompt="System")

        messages = [
            ContextMessage("User message", "alice", datetime.now(), False),
            ContextMessage("Bot response", "brok", datetime.now(), True),
        ]

        result = template._context_messages_to_text(messages)

        assert "alice: User message" in result
        assert "🤖 brok: Bot response" in result

    def test_tool_schemas_to_text_conversion(self):
        """Test conversion of tool schemas to text format."""
        template = LightweightXMLPromptTemplate(system_prompt="System")

        schemas = [
            {"name": "weather", "description": "Get weather info"},
            {"name": "calculator", "description": "Do math"},
        ]

        result = template._tool_schemas_to_text(schemas)

        assert "weather: Get weather info" in result
        assert "calculator: Do math" in result


class TestLightweightXMLHelperFunctions:
    """Test cases for lightweight XML helper functions (KEP-002 Increment D)."""

    def test_get_lightweight_xml_template_concise(self):
        """Test getting lightweight XML template with concise style."""
        template = get_lightweight_xml_template("concise")

        assert isinstance(template, LightweightXMLPromptTemplate)
        assert template.system_prompt == DEFAULT_CONCISE_PROMPT

    def test_get_lightweight_xml_template_detailed(self):
        """Test getting lightweight XML template with detailed style."""
        template = get_lightweight_xml_template("detailed")

        assert isinstance(template, LightweightXMLPromptTemplate)
        assert template.system_prompt == DEFAULT_DETAILED_PROMPT

    def test_get_optimal_xml_template_small_model(self):
        """Test optimal template selection for small models."""
        # Test with 2B model
        template = get_optimal_xml_template("concise", "llama3.2:3b")

        assert isinstance(template, LightweightXMLPromptTemplate)

    def test_get_optimal_xml_template_large_model(self):
        """Test optimal template selection for large models."""
        # Test with larger model
        template = get_optimal_xml_template("concise", "llama3:70b")

        assert isinstance(template, XMLPromptTemplate)
        assert not isinstance(template, LightweightXMLPromptTemplate)

    def test_get_optimal_xml_template_force_lightweight(self):
        """Test forcing lightweight template regardless of model."""
        template = get_optimal_xml_template(
            "concise",
            "gpt-4",  # Large model
            force_lightweight=True,
        )

        assert isinstance(template, LightweightXMLPromptTemplate)

    def test_get_optimal_xml_template_small_model_detection(self):
        """Test automatic detection of various small model patterns."""
        small_models = [
            "tinyllama",
            "gemma-2b",
            "phi-2",
            "stablelm-2b",
            "qwen1.5-1.8b",
            "custom-model-2b",
        ]

        for model_name in small_models:
            template = get_optimal_xml_template("concise", model_name)
            assert isinstance(template, LightweightXMLPromptTemplate), (
                f"Failed to detect {model_name} as small model"
            )


class TestLightweightXMLPerformance:
    """Performance tests for lightweight XML templates (KEP-002 Increment D)."""

    def test_token_overhead_meets_target(self):
        """Test that lightweight XML meets the 20% overhead target."""
        template = LightweightXMLPromptTemplate(system_prompt="Be helpful.")

        # Test with realistic scenario
        context_messages = [
            ContextMessage("How's the weather?", "user1", datetime.now(), False),
            ContextMessage("I'll check for you.", "brok", datetime.now(), True),
        ]

        tool_schemas = [
            {
                "name": "weather",
                "description": "Get current weather",
                "parameters": {
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]

        analysis = template.measure_token_efficiency(
            user_input="What about tomorrow?",
            context_messages=context_messages,
            tool_schemas=tool_schemas,
        )

        # Should meet KEP-002 target
        assert analysis["meets_target"] is True, (
            f"Token overhead {analysis['token_overhead_percent']:.1f}% exceeds 20% target"
        )

    def test_generation_performance_target(self):
        """Test that XML generation meets <5ms performance target."""
        template = LightweightXMLPromptTemplate(system_prompt="Test system")

        # Generate reasonably complex prompt
        context_messages = [
            ContextMessage(f"Message {i}", f"user{i}", datetime.now(), False)
            for i in range(10)
        ]

        analysis = template.measure_token_efficiency(
            user_input="Complex question with context",
            context_messages=context_messages,
        )

        # Should meet KEP-002 performance target
        assert analysis["generation_time_ms"] < 5.0, (
            f"Generation time {analysis['generation_time_ms']:.1f}ms exceeds 5ms target"
        )

    def test_2b_model_recommendation_logic(self):
        """Test recommendation logic for 2B models."""
        template = LightweightXMLPromptTemplate(system_prompt="Short prompt")

        # Simple case that should be recommended for 2B models
        analysis = template.measure_token_efficiency(
            user_input="Simple question",
        )

        # Should be recommended for 2B models if efficient
        if analysis["xml_is_efficient"] and analysis["meets_target"]:
            assert analysis["recommended_for_2b"] is True

    def test_token_efficiency_comparison(self):
        """Test that lightweight XML is more efficient than regular XML."""
        lightweight_template = LightweightXMLPromptTemplate(system_prompt="Be helpful.")
        regular_template = XMLPromptTemplate(system_prompt="Be helpful.")

        # Create same prompt with both templates
        user_input = "What's the weather in London?"
        context_messages = [
            ContextMessage("Previous context", "user1", datetime.now(), False),
        ]

        lightweight_prompt = lightweight_template.build_prompt(
            user_input=user_input,
            xml_formatting=True,
            context_messages=context_messages,
        )

        regular_prompt = regular_template.build_prompt(
            user_input=user_input,
            xml_formatting=True,
            context_messages=context_messages,
        )

        # Lightweight should be more compact
        assert len(lightweight_prompt) <= len(regular_prompt), (
            "Lightweight XML should be more compact than regular XML"
        )


class TestPromptTokenLogging:
    """Test cases for prompt token logging functionality."""

    def test_prompt_template_logging_disabled_by_default(self):
        """Test that token logging is disabled by default."""
        template = PromptTemplate(system_prompt="Test system")

        with patch("brok.prompts.logger") as mock_logger:
            result = template.build_prompt("Hello", log_tokens=False)

            # Should not log when disabled
            mock_logger.log.assert_not_called()
            assert "Hello" in result

    def test_prompt_template_logging_enabled(self):
        """Test that token logging works when enabled."""
        template = PromptTemplate(system_prompt="Test system")

        with patch("brok.prompts.logger") as mock_logger:
            result = template.build_prompt("Hello", log_tokens=True)

            # Should log when enabled
            mock_logger.log.assert_called()
            mock_logger.debug.assert_called()
            assert "Hello" in result

    def test_xml_template_logging_with_overhead_calculation(self):
        """Test XML template logging includes overhead calculation."""
        template = XMLPromptTemplate(system_prompt="Test system")

        with patch("brok.prompts.logger") as mock_logger:
            result = template.build_prompt(
                "Hello", xml_formatting=True, log_tokens=True
            )

            # Should log with XML overhead metrics
            mock_logger.log.assert_called()

            # Check that the log call includes XML overhead
            log_call_args = mock_logger.log.call_args[0]
            log_message = log_call_args[1] if len(log_call_args) > 1 else ""
            assert "xml_overhead=" in log_message
            assert "type=xml" in log_message
            assert "Hello" in result

    def test_lightweight_xml_template_efficiency_logging(self):
        """Test lightweight XML template logs efficiency comparisons."""
        template = LightweightXMLPromptTemplate(system_prompt="Test system")

        with patch("brok.prompts.logger") as mock_logger:
            result = template.build_prompt(
                "Hello", xml_formatting=True, log_tokens=True
            )

            # Should log with efficiency comparison
            mock_logger.log.assert_called()
            mock_logger.info.assert_called()

            # Check for efficiency comparison log
            info_calls = [
                call
                for call in mock_logger.info.call_args_list
                if "XML Efficiency Comparison" in str(call)
            ]
            assert len(info_calls) > 0

            efficiency_log = str(info_calls[0])
            assert "Savings vs Regular:" in efficiency_log
            assert "2B Model Optimized:" in efficiency_log
            assert "Hello" in result

    def test_log_prompt_metrics_performance_levels(self):
        """Test that logging uses appropriate levels based on performance."""
        template = PromptTemplate(system_prompt="Test")

        # Test with different generation times by mocking
        with (
            patch("brok.prompts.time.perf_counter") as mock_time,
            patch("brok.prompts.logger") as mock_logger,
        ):
            # Mock slow generation (>10ms)
            mock_time.side_effect = [0.0, 0.015]  # 15ms

            template._log_prompt_metrics(
                prompt_type="test",
                prompt="test prompt",
                generation_time_ms=15.0,
                user_input="test input",
            )

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

    def test_token_efficiency_metrics_in_logs(self):
        """Test that token efficiency metrics are properly logged."""
        template = LightweightXMLPromptTemplate(system_prompt="Be helpful")

        with patch("brok.prompts.logger") as mock_logger:
            template.build_prompt(
                "What's the weather?", xml_formatting=True, log_tokens=True
            )

            # Should include efficiency metrics
            mock_logger.log.assert_called()

            # Check for token metrics in log
            log_calls = mock_logger.log.call_args_list
            log_messages = [str(call) for call in log_calls]

            # Should have token count and efficiency in logs
            metrics_log = next((msg for msg in log_messages if "tokens=" in msg), "")
            assert "tokens=" in metrics_log
            assert "efficiency=" in metrics_log
            assert "type=lightweight_xml" in metrics_log
