"""Centralized prompt management for the chatbot."""

from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET


@dataclass
class PromptTemplate:
    """Template for building LLM prompts."""

    system_prompt: str
    user_prefix: str = "User"
    assistant_prefix: str = "Assistant"

    def build_prompt(
        self,
        user_input: str,
        context: str | None = None,
        tools_description: str | None = None,
    ) -> str:
        """Build a complete prompt from user input and optional context.

        Args:
            user_input: The user's message
            context: Optional conversation context
            tools_description: Optional description of available tools

        Returns:
            str: The complete formatted prompt
        """
        parts = []

        # Add system prompt if provided
        if self.system_prompt.strip():
            parts.append(f"System: {self.system_prompt}")

        # Add tools description if provided
        if tools_description and tools_description.strip():
            parts.append(f"Tools: {tools_description}")
            parts.append(
                "You can use tools by responding in this format: "
                '{"tool": "tool_name", "params": {"param": "value"}} '
                "OR by using natural language like 'Let me check the weather in London'"
            )

        # Add context if provided
        if context and context.strip():
            parts.append(f"Context:\n{context}")

        # Add user input and assistant prompt
        parts.append(f"{self.user_prefix}: {user_input}")
        parts.append(f"{self.assistant_prefix}:")

        return "\n\n".join(parts)


@dataclass
class XMLPromptTemplate(PromptTemplate):
    """XML-formatted template for building LLM prompts (KEP-002).

    Extends PromptTemplate to provide structured XML formatting while maintaining
    the same API. When used with xml_prompt_formatting=False, produces identical
    output to the base PromptTemplate class.
    """

    def build_prompt(
        self,
        user_input: str,
        context: str | None = None,
        tools_description: str | None = None,
        xml_formatting: bool = False,
    ) -> str:
        """Build a complete prompt with optional XML formatting.

        Args:
            user_input: The user's message
            context: Optional conversation context
            tools_description: Optional description of available tools
            xml_formatting: Whether to use XML structure (KEP-002)

        Returns:
            str: The complete formatted prompt
        """
        if not xml_formatting:
            # When XML formatting is disabled, delegate to parent for identical output
            return super().build_prompt(user_input, context, tools_description)

        return self._build_xml_prompt(user_input, context, tools_description)

    def _build_xml_prompt(
        self,
        user_input: str,
        context: str | None = None,
        tools_description: str | None = None,
    ) -> str:
        """Build XML-formatted prompt structure.

        Creates structured XML with clear semantic boundaries:
        - <system>: System instructions
        - <tools>: Available tools and usage instructions
        - <context>: Conversation history
        - <request>: Current user input and response prompt

        Args:
            user_input: The user's message
            context: Optional conversation context
            tools_description: Optional description of available tools

        Returns:
            str: XML-formatted prompt
        """
        root = ET.Element("prompt", version="1.0")

        # Add system section if provided
        if self.system_prompt.strip():
            system_elem = ET.SubElement(root, "system", role="assistant", name="brok")
            system_elem.text = self.system_prompt.strip()

        # Add tools section if provided
        if tools_description and tools_description.strip():
            tools_elem = ET.SubElement(root, "tools", count="1")

            # Add tools description
            tools_desc_elem = ET.SubElement(tools_elem, "description")
            tools_desc_elem.text = tools_description.strip()

            # Add usage instructions
            usage_elem = ET.SubElement(tools_elem, "usage_instructions")
            usage_elem.text = (
                "You can use tools by responding in this format: "
                '{"tool": "tool_name", "params": {"param": "value"}} '
                "OR by using natural language like 'Let me check the weather in London'"
            )

        # Add context section if provided
        if context and context.strip():
            context_elem = ET.SubElement(root, "context", window_size="10")
            # For now, add context as simple text
            # Future increments will structure individual messages
            context_elem.text = context.strip()

        # Add request section
        request_elem = ET.SubElement(root, "request", sender="user")

        user_input_elem = ET.SubElement(request_elem, "user_input")
        user_input_elem.text = user_input

        response_prompt_elem = ET.SubElement(request_elem, "response_prompt")
        response_prompt_elem.text = f"{self.assistant_prefix}:"

        # Convert to pretty-printed XML string
        return self._format_xml(root)

    def _format_xml(self, root: ET.Element) -> str:
        """Format XML element tree as pretty-printed string.

        Args:
            root: Root XML element

        Returns:
            str: Pretty-printed XML string
        """
        # Set proper indentation manually for clean output
        self._indent_xml(root)
        # Convert to string
        xml_str = ET.tostring(root, encoding="unicode")
        return xml_str

    def _indent_xml(self, elem: ET.Element, level: int = 0) -> None:
        """Add proper indentation to XML elements.

        Args:
            elem: XML element to indent
            level: Current indentation level
        """
        indent = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
            for child in elem:
                self._indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent
        elif level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent


# Default system prompts for different response styles
DEFAULT_CONCISE_PROMPT = """You are brok, a helpful AI assistant in a chat room. Keep responses concise and under 2-3 sentences. Be friendly but brief. Avoid lengthy explanations unless specifically asked. If someone needs detailed information, offer to provide more details if needed."""

DEFAULT_DETAILED_PROMPT = """You are brok, a helpful AI assistant in a chat room. Provide thorough and detailed responses when asked. Be helpful and informative, giving comprehensive answers with examples when appropriate."""

DEFAULT_ADAPTIVE_PROMPT = """You are brok, a helpful AI assistant in a chat room. Adapt your response length to the complexity of the question. For simple questions, keep responses brief (1-2 sentences). For complex topics, provide more detailed explanations as needed."""


# Pre-built templates for common use cases
CONCISE_TEMPLATE = PromptTemplate(
    system_prompt=DEFAULT_CONCISE_PROMPT,
    user_prefix="User",
    assistant_prefix="Assistant",
)

DETAILED_TEMPLATE = PromptTemplate(
    system_prompt=DEFAULT_DETAILED_PROMPT,
    user_prefix="User",
    assistant_prefix="Assistant",
)

ADAPTIVE_TEMPLATE = PromptTemplate(
    system_prompt=DEFAULT_ADAPTIVE_PROMPT,
    user_prefix="User",
    assistant_prefix="Assistant",
)


def get_prompt_template(style: str = "concise") -> PromptTemplate:
    """Get a prompt template by style name.

    Args:
        style: Template style ("concise", "detailed", "adaptive")

    Returns:
        PromptTemplate: The requested template

    Raises:
        ValueError: If style is not recognized
    """
    templates = {
        "concise": CONCISE_TEMPLATE,
        "detailed": DETAILED_TEMPLATE,
        "adaptive": ADAPTIVE_TEMPLATE,
    }

    if style not in templates:
        raise ValueError(
            f"Unknown prompt style: {style}. Available: {list(templates.keys())}"
        )

    return templates[style]


def create_custom_template(system_prompt: str) -> PromptTemplate:
    """Create a custom prompt template with the given system prompt.

    Args:
        system_prompt: Custom system prompt text

    Returns:
        PromptTemplate: A new template with the custom system prompt
    """
    return PromptTemplate(
        system_prompt=system_prompt,
        user_prefix="User",
        assistant_prefix="Assistant",
    )
