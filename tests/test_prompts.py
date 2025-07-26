"""Tests for prompt management functionality."""

import pytest

from brok.prompts import (
    ADAPTIVE_TEMPLATE,
    CONCISE_TEMPLATE,
    DEFAULT_ADAPTIVE_PROMPT,
    DEFAULT_CONCISE_PROMPT,
    DEFAULT_DETAILED_PROMPT,
    DETAILED_TEMPLATE,
    PromptTemplate,
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
