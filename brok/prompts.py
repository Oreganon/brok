"""Centralized prompt management for the chatbot."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from brok.chat import ContextMessage

# Module logger
logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Markdown-structured template for building LLM prompts."""

    system_prompt: str
    user_prefix: str = "User"
    assistant_prefix: str = "Assistant"

    def build_prompt(
        self,
        user_input: str,
        context: str | None = None,
        tools_description: str | None = None,
        xml_formatting: bool = False,  # Ignored - kept for LLM provider compatibility  # noqa: ARG002
        context_messages: list[ContextMessage] | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
        log_tokens: bool = False,
    ) -> str:
        """Build a complete prompt using markdown structure.

        Args:
            user_input: The user's message
            context: Optional conversation context (legacy string format)
            tools_description: Optional description of available tools (legacy format)
            xml_formatting: Ignored (kept for LLM provider compatibility)
            context_messages: Optional structured context messages
            tool_schemas: Optional structured tool schemas
            log_tokens: Whether to log token metrics

        Returns:
            str: The complete markdown-formatted prompt
        """
        start_time = time.perf_counter()

        sections = []

        # Instructions section
        if self.system_prompt.strip():
            sections.append(f"## Instructions\n{self.system_prompt.strip()}")

        # Tools section - prefer structured schemas over description
        if tool_schemas:
            tools_section = self._build_tools_section(tool_schemas)
            sections.append(tools_section)
        elif tools_description and tools_description.strip():
            sections.append(f"## Tools Available\n{tools_description.strip()}")
            sections.append(
                '\n**Usage:** Format as JSON `{"tool": "name", "params": {"key": "value"}}` '
                "or use natural language"
            )

        # Context section - prefer structured messages over string
        if context_messages:
            context_section = self._build_context_section(context_messages)
            sections.append(context_section)
        elif context and context.strip():
            sections.append(f"## Recent Context\n{context.strip()}")

        # Request section
        sections.append(f"## Request\n{user_input}")
        sections.append(f"\n---\n**{self.assistant_prefix}:**")

        prompt = "\n\n".join(sections)

        # Log prompt metrics if requested
        if log_tokens:
            generation_time_ms = (time.perf_counter() - start_time) * 1000
            self._log_prompt_metrics(prompt, generation_time_ms, user_input)

        return prompt

    def _build_tools_section(self, tool_schemas: list[dict[str, Any]]) -> str:
        """Build structured tools section in markdown format."""
        if not tool_schemas:
            return ""

        lines = ["## Tools Available"]

        for schema in tool_schemas:
            name = schema.get("name", "unknown")
            description = schema.get("description", "")
            parameters = schema.get("parameters", {})

            # Tool header
            lines.append(f"- **{name}**: {description}")

            # Parameters if available
            if parameters and parameters.get("properties"):
                properties = parameters.get("properties", {})
                required_params = parameters.get("required", [])

                for param_name, param_info in properties.items():
                    param_type = param_info.get("type", "string")
                    param_desc = param_info.get("description", "")
                    required_marker = (
                        " (required)" if param_name in required_params else ""
                    )

                    lines.append(
                        f"  - `{param_name}` ({param_type}){required_marker}: {param_desc}"
                    )

        # Usage examples
        lines.append("\n**Usage Examples:**")
        if tool_schemas:
            first_tool = tool_schemas[0]
            tool_name = first_tool.get("name", "tool")
            lines.append(
                f'- JSON: `{{"tool": "{tool_name}", "params": {{"key": "value"}}}}`'
            )
            lines.append(f'- Natural: "Use {tool_name} to help me"')

        return "\n".join(lines)

    def _build_context_section(self, context_messages: list[ContextMessage]) -> str:
        """Build structured context section in markdown format."""
        if not context_messages:
            return ""

        lines = ["## Recent Context"]

        for msg in context_messages:
            # Use emoji to distinguish bot vs user messages
            prefix = "🤖" if msg.is_bot else "👤"
            lines.append(f"- {prefix} **{msg.sender}**: {msg.content}")

        return "\n".join(lines)

    def _log_prompt_metrics(
        self, prompt: str, generation_time_ms: float, user_input: str
    ) -> None:
        """Log prompt metrics for performance monitoring."""
        char_count = len(prompt)

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

        logger.log(
            log_level,
            f"Prompt generated [MARKDOWN]: chars={char_count}, gen_time={generation_time_ms:.2f}ms, perf={perf_status}",
        )

        logger.debug(
            f"Prompt details - User input: '{user_input[:50]}{'...' if len(user_input) > 50 else ''}'"
        )


# Single system prompt optimized for concise chat responses
SYSTEM_PROMPT = """You are Brok, a helpful AI assistant in a chat room. When people mention or talk to you, respond directly and concisely in 1-2 sentences. Be friendly but brief.

**Tool Usage:**
- Use tools when they can help answer questions
- Format as JSON: `{"tool": "tool_name", "params": {"param": "value"}}`
- Natural language works too: "Let me check the weather"
- Only use tools that are explicitly available

**Important:** Never mention your own name in responses. Refer to users by their username."""


# Single default template
DEFAULT_TEMPLATE = PromptTemplate(
    system_prompt=SYSTEM_PROMPT,
    user_prefix="User",
    assistant_prefix="Assistant",
)


def get_prompt_template() -> PromptTemplate:
    """Get the default prompt template.

    Returns:
        PromptTemplate: The default template
    """
    return DEFAULT_TEMPLATE


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
