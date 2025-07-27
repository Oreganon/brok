"""Centralized prompt management for the chatbot."""

from __future__ import annotations

from dataclasses import dataclass
import json

# Import for KEP-002 Increment B structured context
from typing import TYPE_CHECKING, Any
import xml.etree.ElementTree as ET

if TYPE_CHECKING:
    from brok.chat import ContextMessage


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
        context_messages: list[ContextMessage] | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> str:
        """Build a complete prompt with optional XML formatting.

        Args:
            user_input: The user's message
            context: Optional conversation context (legacy string format)
            tools_description: Optional description of available tools (legacy format)
            xml_formatting: Whether to use XML structure (KEP-002)
            context_messages: Optional structured context messages (KEP-002 Increment B)
            tool_schemas: Optional structured tool schemas (KEP-002 Increment C)

        Returns:
            str: The complete formatted prompt
        """
        if not xml_formatting:
            # When XML formatting is disabled, delegate to parent for identical output
            return super().build_prompt(user_input, context, tools_description)

        return self._build_xml_prompt(
            user_input, context, tools_description, context_messages, tool_schemas
        )

    def _build_xml_prompt(
        self,
        user_input: str,
        context: str | None = None,
        tools_description: str | None = None,
        context_messages: list[ContextMessage] | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> str:
        """Build XML-formatted prompt structure.

        Creates structured XML with clear semantic boundaries:
        - <instructions>: System instructions and response guidelines
        - <tools>: Available tools and usage instructions
        - <context>: Conversation history
        - <request>: Current user input and response prompt

        Args:
            user_input: The user's message
            context: Optional conversation context (legacy string format)
            tools_description: Optional description of available tools (legacy format)
            context_messages: Optional structured context messages (KEP-002 Increment B)
            tool_schemas: Optional structured tool schemas (KEP-002 Increment C)

        Returns:
            str: XML-formatted prompt
        """
        root = ET.Element("prompt", version="1.0")

        # Add system section if provided
        if self.system_prompt.strip():
            system_elem = ET.SubElement(
                root, "instructions", role="assistant", name="brok"
            )
            # Add XML response guidance to prevent XML bleeding in responses
            enhanced_prompt = self.system_prompt.strip()
            enhanced_prompt += "\n\nIMPORTANT: Respond only in plain text. Do not include any XML tags or markup in your response."
            system_elem.text = enhanced_prompt

        # Add tools section if provided (KEP-002 Increment C)
        if tool_schemas:
            # Structured tools with individual <tool> elements (KEP-002 Increment C)
            self._add_structured_tools(root, tool_schemas)
        elif tools_description and tools_description.strip():
            # Legacy tools format for backward compatibility
            tools_elem = ET.SubElement(root, "tools", count="1", format="legacy")

            # Add enhanced tools description
            tools_desc_elem = ET.SubElement(tools_elem, "description")
            tools_desc_elem.text = tools_description.strip()

            # Add comprehensive usage instructions
            usage_elem = ET.SubElement(tools_elem, "usage_instructions")
            usage_elem.text = (
                "TOOL USAGE GUIDELINES:\n"
                '• Use JSON format for reliability: {"tool": "tool_name", "params": {"key": "value"}}\n'
                '• Natural language also works: "Let me check the weather in London"\n'
                "• Only use tools that are explicitly listed in the description above\n"
                "• Always include required parameters and validate optional ones\n"
                "• If a tool doesn't exist, apologize and suggest available alternatives\n"
                "• Never attempt to use tools not documented above"
            )

            # Add validation requirements
            validation_elem = ET.SubElement(tools_elem, "validation_requirements")
            validation_elem.text = (
                "Before using any tool:\n"
                "1. Verify the tool exists in the tools description\n"
                "2. Check all required parameters are provided\n"
                "3. Validate parameter types match expectations\n"
                "4. If unsure about usage, ask for clarification\n"
                "5. Handle errors gracefully and suggest alternatives"
            )

        # Add context section if provided (KEP-002 Increment B)
        if context_messages:
            # Structured context with individual message elements (KEP-002 Increment B)
            self._add_structured_context(root, context_messages)
        elif context and context.strip():
            # Legacy string context for backward compatibility
            context_elem = ET.SubElement(
                root, "context", window_size="10", format="legacy"
            )
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

    def _add_structured_context(
        self, root: ET.Element, context_messages: list[ContextMessage]
    ) -> None:
        """Add structured context section with individual message elements.

        Creates XML context with individual <message> elements containing
        sender, timestamp, type attributes and priority-based ordering.

        Args:
            root: Root XML element to add context to
            context_messages: List of structured context messages
        """
        if not context_messages:
            return

        # Create context element with metadata
        context_elem = ET.SubElement(
            root, "context", window_size=str(len(context_messages)), format="structured"
        )

        # Add individual message elements with metadata
        for ctx_msg in context_messages:
            # Determine message type
            msg_type = "bot_response" if ctx_msg.is_bot else "user_message"

            # Create message element with attributes
            message_elem = ET.SubElement(
                context_elem,
                "message",
                sender=ctx_msg.sender,
                timestamp=ctx_msg.timestamp.isoformat(),
                type=msg_type,
                id=ctx_msg.message_id,
            )

            # Set message content
            message_elem.text = ctx_msg.content

    def _add_structured_tools(
        self, root: ET.Element, tool_schemas: list[dict[str, Any]]
    ) -> None:
        """Add structured tools section with individual tool elements.

        Creates XML tools with individual <tool> elements containing
        parameter definitions, examples, and usage instructions.

        Args:
            root: Root XML element to add tools to
            tool_schemas: List of tool schema dictionaries
        """
        if not tool_schemas:
            return

        # Create tools element with metadata
        tools_elem = ET.SubElement(
            root, "tools", count=str(len(tool_schemas)), format="structured"
        )

        # Add tool usage guidelines at the top level
        guidelines_elem = ET.SubElement(tools_elem, "usage_guidelines")
        guidelines_elem.text = (
            "IMPORTANT TOOL USAGE RULES:\n"
            "• Only use tools that are explicitly listed below\n"
            '• Use JSON format: {"tool": "tool_name", "params": {"key": "value"}}\n'
            '• Natural language also works: "Let me check the weather in London"\n'
            "• Always include required parameters and validate optional ones\n"
            "• If a tool doesn't exist, apologize and suggest available alternatives\n"
            "• Never attempt to use tools not in this list"
        )

        for tool_schema in tool_schemas:
            tool_name = tool_schema.get("name", "unknown")
            description = tool_schema.get("description", "")
            parameters = tool_schema.get("parameters", {})

            # Create individual tool element
            tool_elem = ET.SubElement(
                tools_elem,
                "tool",
                name=tool_name,
                category="function",
                available="true",
            )

            # Add description
            desc_elem = ET.SubElement(tool_elem, "description")
            desc_elem.text = description

            # Add parameters section if available
            if parameters:
                params_elem = ET.SubElement(tool_elem, "parameters")

                # Add parameter definitions
                properties = parameters.get("properties", {})
                required_params = parameters.get("required", [])

                for param_name, param_schema in properties.items():
                    param_elem = ET.SubElement(
                        params_elem,
                        "parameter",
                        name=param_name,
                        type=param_schema.get("type", "string"),
                        required=str(param_name in required_params).lower(),
                    )

                    # Set parameter description
                    param_elem.text = param_schema.get("description", "")

                    # Add examples if available
                    examples = param_schema.get("examples", [])
                    if examples:
                        examples_elem = ET.SubElement(param_elem, "examples")
                        for example in examples[:3]:  # Limit to 3 examples
                            example_elem = ET.SubElement(examples_elem, "example")
                            example_elem.text = str(example)

                # Add usage examples section
                examples_elem = ET.SubElement(tool_elem, "usage_examples")

                # JSON example
                json_example_elem = ET.SubElement(examples_elem, "json_format")
                json_example = self._generate_json_example(tool_name, parameters)
                json_example_elem.text = json_example

                # Natural language examples
                nl_examples = self._generate_natural_language_examples(tool_name)
                if nl_examples:
                    nl_elem = ET.SubElement(examples_elem, "natural_language")
                    for example in nl_examples[:2]:  # Limit to 2 examples
                        example_elem = ET.SubElement(nl_elem, "example")
                        example_elem.text = example

        # Add validation rules at the end
        validation_elem = ET.SubElement(tools_elem, "validation_rules")
        validation_elem.text = (
            "Before using any tool:\n"
            "1. Verify the tool name exists in the list above\n"
            "2. Check that all required parameters are provided\n"
            "3. Validate parameter types and formats\n"
            "4. If unsure, ask for clarification rather than guessing\n"
            "5. If a tool fails, explain what went wrong and suggest alternatives"
        )

    def _generate_json_example(self, tool_name: str, parameters: dict[str, Any]) -> str:
        """Generate a JSON usage example for a tool."""
        example_params: dict[str, Any] = {}
        properties = parameters.get("properties", {})

        # Generate realistic example values for parameters
        for param_name, param_schema in properties.items():
            param_type = param_schema.get("type", "string")
            examples = param_schema.get("examples", [])

            if examples:
                # Use the first example if available
                example_params[param_name] = examples[0]
            elif param_type == "string":
                # Use parameter-specific examples
                if "city" in param_name.lower():
                    example_params[param_name] = "London"
                elif "expression" in param_name.lower():
                    example_params[param_name] = "2 + 3 * 4"
                elif "timezone" in param_name.lower():
                    example_params[param_name] = "UTC"
                elif "format" in param_name.lower():
                    example_params[param_name] = "readable"
                else:
                    example_params[param_name] = "example_value"
            elif param_type == "number":
                example_params[param_name] = 42
            elif param_type == "boolean":
                example_params[param_name] = True

        return json.dumps(
            {"tool": tool_name, "params": example_params}, separators=(", ", ": ")
        )

    def _generate_natural_language_examples(self, tool_name: str) -> list[str]:
        """Generate natural language usage examples for a tool."""
        examples = []

        if tool_name == "weather":
            examples = [
                "What's the weather like in London?",
                "Check the weather in New York for me",
            ]
        elif tool_name == "calculator":
            examples = ["Calculate 2 + 3 * 4", "What's the square root of 16?"]
        elif tool_name == "datetime":
            examples = ["What time is it?", "What's the current time in Tokyo?"]
        else:
            # Generic examples
            examples = [f"Use the {tool_name} tool", f"Help me with {tool_name}"]

        return examples


# Default system prompts for different response styles
DEFAULT_CONCISE_PROMPT = """You are Brok, a helpful AI assistant in a chat room. When people mention or talk to you, respond to them directly and concisely in 2-3 sentences. Be friendly but brief. Avoid lengthy explanations unless specifically asked. If someone needs detailed information, offer to provide more details if needed.

TOOL USAGE GUIDELINES:
- Use available tools when they can help answer user questions
- Only use tools that are explicitly listed in your tool documentation
- Format tool calls as JSON: {"tool": "tool_name", "params": {"parameter": "value"}}
- You can also use natural language that triggers tool detection
- If a tool doesn't exist, apologize and suggest what you can do instead
- Always validate tool parameters before using them

IMPORTANT: Never mention your own name in responses. Always refer to the people who are talking to you by username. Focus on helping the person who messaged you."""

DEFAULT_DETAILED_PROMPT = """You are Brok, a helpful AI assistant in a chat room. When people mention or talk to you, provide thorough and detailed responses to help them. Be helpful and informative, giving comprehensive answers with examples when appropriate.

TOOL USAGE GUIDELINES:
- Use available tools whenever they can enhance your response with real-time or computed information
- Only use tools that are explicitly documented as available to you
- Format tool calls as JSON: {"tool": "tool_name", "params": {"parameter": "value"}}
- You can also phrase requests naturally (e.g., "Let me check the weather in London")
- Always check that required parameters are provided and valid
- If a requested tool doesn't exist, explain what tools you do have access to
- Provide context about what the tool does when using it

IMPORTANT: Never mention your own name in responses. Always refer to the people who are talking to you by username. Focus on helping the person who messaged you."""

DEFAULT_ADAPTIVE_PROMPT = """You are Brok, a helpful AI assistant in a chat room. When people mention or talk to you, adapt your response length to the complexity of their question. For simple questions, keep responses brief (1-2 sentences). For complex topics, provide more detailed explanations as needed.

TOOL USAGE GUIDELINES:
- Use tools when they add value to your response (weather, calculations, time, etc.)
- Only use tools that are listed in your available tools documentation
- Prefer JSON format for reliability: {"tool": "tool_name", "params": {"key": "value"}}
- Natural language works too: "Let me calculate that" or "Let me check the weather"
- Never attempt to use tools that aren't in your available tools list
- If asked about unavailable tools, explain what tools you can use instead
- Always include required parameters and validate optional ones

IMPORTANT: Never mention your own name in responses. Always refer to the people who are talking to you by username. Focus on helping the person who messaged you."""


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


def get_xml_prompt_template(style: str = "concise") -> XMLPromptTemplate:
    """Get an XML prompt template by style name (KEP-002 Increment B).

    Args:
        style: Template style ("concise", "detailed", "adaptive")

    Returns:
        XMLPromptTemplate: The requested XML template

    Raises:
        ValueError: If style is not recognized
    """
    # Get the corresponding base template
    base_template = get_prompt_template(style)

    # Create XMLPromptTemplate with same configuration
    return XMLPromptTemplate(
        system_prompt=base_template.system_prompt,
        user_prefix=base_template.user_prefix,
        assistant_prefix=base_template.assistant_prefix,
    )


def create_custom_xml_template(system_prompt: str) -> XMLPromptTemplate:
    """Create a custom XML prompt template with the given system prompt (KEP-002 Increment B).

    Args:
        system_prompt: Custom system prompt text

    Returns:
        XMLPromptTemplate: A new XML template with the custom system prompt
    """
    return XMLPromptTemplate(
        system_prompt=system_prompt,
        user_prefix="User",
        assistant_prefix="Assistant",
    )
