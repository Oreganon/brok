"""Centralized prompt management for the chatbot."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import time

# Import for KEP-002 Increment B structured context
from typing import TYPE_CHECKING, Any
import xml.etree.ElementTree as ET

# Import for KEP-002 Increment D token optimization
from brok.token_counter import get_token_counter

if TYPE_CHECKING:
    from brok.chat import ContextMessage

# Module logger
logger = logging.getLogger(__name__)


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
        xml_formatting: bool = False,  # For compatibility with subclasses  # noqa: ARG002
        context_messages: list[ContextMessage] | None = None,  # For compatibility  # noqa: ARG002
        tool_schemas: list[dict[str, Any]] | None = None,  # For compatibility  # noqa: ARG002
        log_tokens: bool = False,
    ) -> str:
        """Build a complete prompt from user input and optional context.

        Args:
            user_input: The user's message
            context: Optional conversation context
            tools_description: Optional description of available tools
            xml_formatting: Ignored in base class (for subclass compatibility)
            context_messages: Ignored in base class (for subclass compatibility)
            tool_schemas: Ignored in base class (for subclass compatibility)
            log_tokens: Whether to log token metrics (for performance monitoring)

        Returns:
            str: The complete formatted prompt
        """
        start_time = time.perf_counter()

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

        prompt = "\n\n".join(parts)

        # Log prompt metrics if requested
        if log_tokens:
            generation_time_ms = (time.perf_counter() - start_time) * 1000
            self._log_prompt_metrics(
                prompt_type="text",
                prompt=prompt,
                generation_time_ms=generation_time_ms,
                user_input=user_input,
                has_context=bool(context and context.strip()),
                has_tools=bool(tools_description and tools_description.strip()),
            )

        return prompt

    def _log_prompt_metrics(
        self,
        prompt_type: str,
        prompt: str,
        generation_time_ms: float,
        user_input: str,
        has_context: bool = False,
        has_tools: bool = False,
        xml_overhead_pct: float | None = None,
    ) -> None:
        """Log prompt metrics for performance monitoring.

        Args:
            prompt_type: Type of prompt (text, xml, lightweight_xml)
            prompt: The generated prompt
            generation_time_ms: Time taken to generate prompt
            user_input: Original user input
            has_context: Whether context was included
            has_tools: Whether tools were included
            xml_overhead_pct: XML overhead percentage (if applicable)
        """
        char_count = len(prompt)

        # Get token count efficiently
        token_counter = get_token_counter()
        token_measurement = token_counter.count_tokens(prompt)

        # Determine log level based on performance
        if generation_time_ms > 10:  # > 10ms
            log_level = logging.WARNING
            perf_status = "SLOW"
        elif generation_time_ms > 5:  # > 5ms
            log_level = logging.INFO
            perf_status = "MODERATE"
        else:
            log_level = logging.DEBUG
            perf_status = "FAST"

        # Base metrics
        metrics_parts = [
            f"type={prompt_type}",
            f"chars={char_count}",
            f"tokens={token_measurement.token_count}",
            f"efficiency={token_measurement.efficiency_ratio:.3f}",
            f"gen_time={generation_time_ms:.2f}ms",
        ]

        # Optional metrics
        if has_context:
            metrics_parts.append("context=yes")
        if has_tools:
            metrics_parts.append("tools=yes")
        if xml_overhead_pct is not None:
            metrics_parts.append(f"xml_overhead={xml_overhead_pct:.1f}%")

        # Performance status
        metrics_parts.append(f"perf={perf_status}")

        logger.log(
            log_level,
            f"Prompt generated [{prompt_type.upper()}]: {', '.join(metrics_parts)}",
        )

        # Additional debug info for development
        logger.debug(
            f"Prompt details - User input: '{user_input[:50]}{'...' if len(user_input) > 50 else ''}', "
            f"Encoding: {token_measurement.encoding_used}, "
            f"Token measurement time: {token_measurement.measurement_time_ms:.2f}ms"
        )


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
        log_tokens: bool = False,
    ) -> str:
        """Build a complete prompt with optional XML formatting.

        Args:
            user_input: The user's message
            context: Optional conversation context (legacy string format)
            tools_description: Optional description of available tools (legacy format)
            xml_formatting: Whether to use XML structure (KEP-002)
            context_messages: Optional structured context messages (KEP-002 Increment B)
            tool_schemas: Optional structured tool schemas (KEP-002 Increment C)
            log_tokens: Whether to log token metrics (for performance monitoring)

        Returns:
            str: The complete formatted prompt
        """
        start_time = time.perf_counter()

        if not xml_formatting:
            # When XML formatting is disabled, delegate to parent for identical output
            return super().build_prompt(
                user_input, context, tools_description, log_tokens
            )

        xml_prompt = self._build_xml_prompt(
            user_input, context, tools_description, context_messages, tool_schemas
        )

        # Log XML prompt metrics if requested
        if log_tokens:
            generation_time_ms = (time.perf_counter() - start_time) * 1000

            # Calculate XML overhead by comparing to text version
            text_prompt = super().build_prompt(
                user_input, context, tools_description, False
            )
            xml_overhead_pct = (
                (len(xml_prompt) - len(text_prompt)) / len(text_prompt)
            ) * 100

            self._log_prompt_metrics(
                prompt_type="xml",
                prompt=xml_prompt,
                generation_time_ms=generation_time_ms,
                user_input=user_input,
                has_context=bool((context and context.strip()) or context_messages),
                has_tools=bool(
                    (tools_description and tools_description.strip()) or tool_schemas
                ),
                xml_overhead_pct=xml_overhead_pct,
            )

        return xml_prompt

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


@dataclass
class LightweightXMLPromptTemplate(XMLPromptTemplate):
    """Optimized XML template for 2B parameter models (KEP-002 Increment D).

    Provides extremely concise XML formatting to minimize token overhead
    for smaller models while maintaining structured benefits.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize with token counter for optimization."""
        super().__init__(*args, **kwargs)
        self._token_counter = get_token_counter()

    def build_prompt(
        self,
        user_input: str,
        context: str | None = None,
        tools_description: str | None = None,
        xml_formatting: bool = False,
        context_messages: list[ContextMessage] | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
        log_tokens: bool = False,
    ) -> str:
        """Build prompt with enhanced logging for 2B model optimization metrics.

        Args:
            user_input: The user's message
            context: Optional conversation context (legacy string format)
            tools_description: Optional description of available tools (legacy format)
            xml_formatting: Whether to use XML structure (KEP-002)
            context_messages: Optional structured context messages (KEP-002 Increment B)
            tool_schemas: Optional structured tool schemas (KEP-002 Increment C)
            log_tokens: Whether to log token metrics (for performance monitoring)

        Returns:
            str: The complete formatted prompt
        """
        start_time = time.perf_counter()

        if not xml_formatting:
            # Delegate to parent when XML disabled
            return super().build_prompt(
                user_input,
                context,
                tools_description,
                xml_formatting,
                context_messages,
                tool_schemas,
                log_tokens,
            )

        lightweight_prompt = self._build_xml_prompt(
            user_input, context, tools_description, context_messages, tool_schemas
        )

        # Enhanced logging for lightweight XML with efficiency comparisons
        if log_tokens:
            generation_time_ms = (time.perf_counter() - start_time) * 1000

            # Get comparison prompts for efficiency analysis
            text_prompt = super().build_prompt(
                user_input, context, tools_description, False
            )

            # Create regular XML for comparison (instantiate XMLPromptTemplate)
            regular_xml_template = XMLPromptTemplate(
                system_prompt=self.system_prompt,
                user_prefix=self.user_prefix,
                assistant_prefix=self.assistant_prefix,
            )
            regular_xml_prompt = regular_xml_template._build_xml_prompt(
                user_input, context, tools_description, context_messages, tool_schemas
            )

            # Calculate efficiency metrics
            text_len = len(text_prompt)
            regular_xml_len = len(regular_xml_prompt)
            lightweight_len = len(lightweight_prompt)

            text_overhead_pct = ((lightweight_len - text_len) / text_len) * 100
            xml_efficiency_pct = (
                (regular_xml_len - lightweight_len) / regular_xml_len
            ) * 100

            # Enhanced logging with comparison metrics
            self._log_prompt_metrics(
                prompt_type="lightweight_xml",
                prompt=lightweight_prompt,
                generation_time_ms=generation_time_ms,
                user_input=user_input,
                has_context=bool((context and context.strip()) or context_messages),
                has_tools=bool(
                    (tools_description and tools_description.strip()) or tool_schemas
                ),
                xml_overhead_pct=text_overhead_pct,
            )

            # Additional efficiency comparison log
            logger.info(
                f"XML Efficiency Comparison - "
                f"Text: {text_len} chars, "
                f"Regular XML: {regular_xml_len} chars, "
                f"Lightweight XML: {lightweight_len} chars | "
                f"Savings vs Regular: {xml_efficiency_pct:.1f}%, "
                f"Overhead vs Text: {text_overhead_pct:.1f}% | "
                f"2B Model Optimized: {lightweight_len < 400 and text_overhead_pct < 50}"
            )

        return lightweight_prompt

    def _build_xml_prompt(
        self,
        user_input: str,
        context: str | None = None,
        tools_description: str | None = None,
        context_messages: list[ContextMessage] | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> str:
        """Build extremely lightweight XML-formatted prompt for 2B models.

        Optimizes for minimal token overhead while preserving structure.
        """
        root = ET.Element("prompt")

        # Minimal system section - no verbose instructions
        if self.system_prompt.strip():
            system_elem = ET.SubElement(root, "system")
            # Strip verbose XML warnings for 2B models - they understand structure
            system_elem.text = self.system_prompt.strip()

        # Ultra-compact tools section
        if tool_schemas:
            self._add_compact_tools(root, tool_schemas)
        elif tools_description and tools_description.strip():
            tools_elem = ET.SubElement(root, "tools")
            tools_elem.text = tools_description.strip()

        # Minimal context section
        if context_messages:
            self._add_compact_context(root, context_messages)
        elif context and context.strip():
            context_elem = ET.SubElement(root, "context")
            context_elem.text = context.strip()

        # Minimal request section
        request_elem = ET.SubElement(root, "request")
        request_elem.text = f"{user_input}\n{self.assistant_prefix}:"

        return self._format_compact_xml(root)

    def _add_compact_tools(
        self, root: ET.Element, tool_schemas: list[dict[str, Any]]
    ) -> None:
        """Add ultra-compact tools section for 2B models."""
        tools_elem = ET.SubElement(root, "tools")

        for tool_schema in tool_schemas:
            tool_name = tool_schema.get("name", "unknown")
            description = tool_schema.get("description", "")
            parameters = tool_schema.get("parameters", {})

            # Minimal tool element - just name and description
            tool_elem = ET.SubElement(tools_elem, "tool", name=tool_name)
            tool_elem.text = description

            # Only add parameters if they're complex
            if parameters and parameters.get("properties"):
                params_text = self._compact_params_text(parameters)
                if params_text:
                    tool_elem.text += f" Params: {params_text}"

    def _compact_params_text(self, parameters: dict[str, Any]) -> str:
        """Generate ultra-compact parameter description."""
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])

        param_parts = []
        for param_name, param_schema in properties.items():
            param_type = param_schema.get("type", "str")
            is_required = param_name in required
            required_marker = "*" if is_required else ""
            param_parts.append(f"{param_name}{required_marker}:{param_type}")

        return "{" + ", ".join(param_parts) + "}"

    def _add_compact_context(
        self, root: ET.Element, context_messages: list[ContextMessage]
    ) -> None:
        """Add ultra-compact context section for 2B models."""
        if not context_messages:
            return

        context_elem = ET.SubElement(root, "context")

        for msg in context_messages:
            # Ultra-minimal message format
            msg_elem = ET.SubElement(
                context_elem,
                "msg",
                u=msg.sender,  # Ultra-short attribute names
                t="b" if msg.is_bot else "u",  # b=bot, u=user
            )
            msg_elem.text = msg.content

    def _format_compact_xml(self, root: ET.Element) -> str:
        """Format XML with minimal whitespace for 2B models."""
        # No indentation to save tokens
        xml_str = ET.tostring(root, encoding="unicode")
        return xml_str

    def measure_token_efficiency(
        self,
        user_input: str,
        context_messages: list[ContextMessage] | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Measure token efficiency compared to text prompt.

        Returns detailed analysis of token overhead for optimization.
        """
        # Generate both XML and text versions
        xml_prompt = self.build_prompt(
            user_input=user_input,
            xml_formatting=True,
            context_messages=context_messages,
            tool_schemas=tool_schemas,
        )

        text_prompt = self.build_prompt(
            user_input=user_input,
            xml_formatting=False,
            context=self._context_messages_to_text(context_messages)
            if context_messages
            else None,
            tools_description=self._tool_schemas_to_text(tool_schemas)
            if tool_schemas
            else None,
        )

        # Measure overhead
        overhead_analysis = self._token_counter.measure_prompt_overhead(
            xml_prompt, text_prompt
        )

        # Add 2B model specific metrics
        xml_measurement = self._token_counter.count_tokens(xml_prompt)
        text_measurement = self._token_counter.count_tokens(text_prompt)

        overhead_analysis.update(
            {
                "xml_is_efficient": xml_measurement.is_efficient,
                "text_is_efficient": text_measurement.is_efficient,
                "generation_time_ms": xml_measurement.measurement_time_ms,
                "recommended_for_2b": (
                    overhead_analysis["meets_target"]
                    and xml_measurement.is_efficient
                    and xml_measurement.token_count
                    < 400  # Conservative limit for 2B models
                ),
            }
        )

        return overhead_analysis

    def _context_messages_to_text(self, messages: list[ContextMessage]) -> str:
        """Convert structured messages to text format for comparison."""
        if not messages:
            return ""

        text_parts = []
        for msg in messages:
            prefix = "🤖 " if msg.is_bot else ""
            text_parts.append(f"{prefix}{msg.sender}: {msg.content}")

        return "\n".join(text_parts)

    def _tool_schemas_to_text(self, schemas: list[dict[str, Any]]) -> str:
        """Convert tool schemas to text format for comparison."""
        if not schemas:
            return ""

        text_parts = []
        for schema in schemas:
            name = schema.get("name", "unknown")
            description = schema.get("description", "")
            text_parts.append(f"{name}: {description}")

        return "\n".join(text_parts)


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


def get_lightweight_xml_template(
    style: str = "concise",
) -> LightweightXMLPromptTemplate:
    """Get a lightweight XML prompt template optimized for 2B models (KEP-002 Increment D).

    Args:
        style: Template style ("concise", "detailed", "adaptive")

    Returns:
        LightweightXMLPromptTemplate: Optimized template for smaller models
    """
    base_template = get_prompt_template(style)

    return LightweightXMLPromptTemplate(
        system_prompt=base_template.system_prompt,
        user_prefix=base_template.user_prefix,
        assistant_prefix=base_template.assistant_prefix,
    )


def create_custom_lightweight_xml_template(
    system_prompt: str,
) -> LightweightXMLPromptTemplate:
    """Create a custom lightweight XML template for 2B models (KEP-002 Increment D).

    Args:
        system_prompt: Custom system prompt text

    Returns:
        LightweightXMLPromptTemplate: Custom optimized template
    """
    return LightweightXMLPromptTemplate(
        system_prompt=system_prompt,
        user_prefix="User",
        assistant_prefix="Assistant",
    )


def get_optimal_xml_template(
    style: str = "concise",
    model_name: str = "llama3.2:3b",
    force_lightweight: bool = False,
) -> XMLPromptTemplate | LightweightXMLPromptTemplate:
    """Get the optimal XML template based on model characteristics.

    Automatically selects between regular XML and lightweight XML based on
    model size and capabilities.

    Args:
        style: Template style ("concise", "detailed", "adaptive")
        model_name: Name of the model being used
        force_lightweight: Force lightweight template regardless of model

    Returns:
        XMLPromptTemplate: Optimal template for the model
    """
    # Detect 2B models that benefit from lightweight formatting
    small_model_indicators = [
        "1b",
        "2b",
        "3b",  # Parameter counts
        "tinyllama",
        "gemma-2b",
        "qwen1.5-1.8b",  # Specific small models
        "phi-",
        "stablelm-2b",  # Model families known to be small
    ]

    is_small_model = force_lightweight or any(
        indicator in model_name.lower() for indicator in small_model_indicators
    )

    if is_small_model:
        return get_lightweight_xml_template(style)
    else:
        return get_xml_prompt_template(style)
