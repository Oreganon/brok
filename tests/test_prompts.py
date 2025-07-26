"""Tests for prompt management functionality."""

from datetime import datetime
import xml.etree.ElementTree as ET

import pytest

from brok.chat import ContextMessage
from brok.prompts import (
    ADAPTIVE_TEMPLATE,
    CONCISE_TEMPLATE,
    DEFAULT_ADAPTIVE_PROMPT,
    DEFAULT_CONCISE_PROMPT,
    DEFAULT_DETAILED_PROMPT,
    DETAILED_TEMPLATE,
    PromptTemplate,
    XMLPromptTemplate,
    create_custom_template,
    get_prompt_template,
)


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

    def test_all_prompts_mention_brok(self):
        """Test that all default prompts identify the bot as 'brok'."""
        prompts = [
            DEFAULT_CONCISE_PROMPT,
            DEFAULT_DETAILED_PROMPT,
            DEFAULT_ADAPTIVE_PROMPT,
        ]

        for prompt in prompts:
            assert "brok" in prompt.lower()
            assert "chat room" in prompt.lower()


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
        assert '<system role="assistant" name="brok">You are brok</system>' in result
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
        assert '<system role="assistant" name="brok">You are helpful</system>' in result
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
        assert '<tools count="1">' in result
        assert '<tool name="weather">' in result
        assert "<description>Get current weather for a city</description>" in result
        assert "<parameters>" in result
        assert '<parameter name="city" type="string" required="true">' in result
        assert "City name for weather lookup" in result

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
        assert '<tools count="2">' in result
        assert '<tool name="weather">' in result
        assert '<tool name="calculator">' in result
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
