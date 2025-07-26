"""Abstract base class and types for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from brok.tools.registry import ToolRegistry


@dataclass
class LLMConfig:
    """Base configuration for LLM providers."""

    model_name: str
    max_tokens: int = 150
    temperature: float = 0.7
    timeout_seconds: int = 30


class LLMMetadata(TypedDict, total=False):
    """Lightweight metadata for LLM responses."""

    tokens_used: int
    provider: str
    model: str


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    This class defines the interface that all LLM providers must implement.
    Providers should handle their own connection management, error handling,
    and response streaming.

    Example:
        >>> class MyProvider(LLMProvider):
        ...     async def generate(self, prompt, context=None):
        ...         # Implementation here
        ...         yield "Hello, world!"
        ...
        ...     async def health_check(self):
        ...         return True
    """

    @abstractmethod
    async def generate(
        self, prompt: str, context: str | None = None
    ) -> AsyncGenerator[str]:
        """Generate response as async generator.

        Yields incremental text chunks for streaming, or final response.
        Callers wanting full response: ''.join([chunk async for chunk in generate(...)])

        Args:
            prompt: The user's input message
            context: Optional conversation context (recent chat history)

        Yields:
            str: Text chunks of the generated response

        Raises:
            LLMConnectionError: When unable to connect to the LLM service
            LLMTimeoutError: When the request times out
            LLMGenerationError: When the LLM fails to generate a response

        Example:
            >>> async for chunk in provider.generate("Hello"):
            ...     print(chunk, end="")
            Hello, how can I help you today?
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is healthy and responsive.

        This method should perform a quick connectivity test without
        generating actual content. Used for monitoring and failover.

        Returns:
            bool: True if provider is available and working

        Raises:
            LLMConnectionError: When unable to connect to the LLM service

        Example:
            >>> if await provider.health_check():
            ...     print("Provider is healthy")
            ... else:
            ...     print("Provider is down")
        """

    def get_metadata(self) -> LLMMetadata:
        """Get metadata from last generation (optional).

        Providers may override this to return useful information about
        the last generation request, such as token usage or model info.

        Returns:
            LLMMetadata: Information about the last generation

        Example:
            >>> metadata = provider.get_metadata()
            >>> print(f"Used {metadata.get('tokens_used', 0)} tokens")
        """
        return {}

    def set_tool_registry(self, registry: ToolRegistry) -> None:
        """Set the tool registry for this LLM provider.

        Args:
            registry: Tool registry instance
        """
        self.tool_registry = registry

    def has_tools(self) -> bool:
        """Check if this provider has tools available.

        Returns:
            bool: True if tools are available
        """
        return hasattr(self, "tool_registry") and self.tool_registry is not None

    def get_tools_description(self) -> str:
        """Get description of available tools for prompt inclusion.

        Returns:
            str: Description of available tools, empty string if none
        """
        if self.has_tools():
            return self.tool_registry.get_tools_description()
        return ""

    def get_tools_schema(self) -> list[dict[str, Any]]:
        """Get structured schema for available tools (KEP-002 Increment C).

        Returns:
            list[dict]: List of tool schemas for structured XML prompts
        """
        if self.has_tools():
            return self.tool_registry.get_tools_schema()
        return []
