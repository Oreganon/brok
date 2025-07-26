"""Centralized prompt management for the chatbot."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Template for building LLM prompts."""

    system_prompt: str
    user_prefix: str = "User"
    assistant_prefix: str = "Assistant"

    def build_prompt(self, user_input: str, context: str | None = None) -> str:
        """Build a complete prompt from user input and optional context.

        Args:
            user_input: The user's message
            context: Optional conversation context

        Returns:
            str: The complete formatted prompt
        """
        parts = []

        # Add system prompt if provided
        if self.system_prompt.strip():
            parts.append(f"System: {self.system_prompt}")

        # Add context if provided
        if context and context.strip():
            parts.append(f"Context:\n{context}")

        # Add user input and assistant prompt
        parts.append(f"{self.user_prefix}: {user_input}")
        parts.append(f"{self.assistant_prefix}:")

        return "\n\n".join(parts)


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
