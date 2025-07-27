"""LlamaCpp HTTP server provider implementation."""

from __future__ import annotations

import asyncio  # Added to catch asyncio.TimeoutError
import logging
from typing import TYPE_CHECKING

import aiohttp

from brok.exceptions import LLMConnectionError, LLMGenerationError, LLMTimeoutError
from brok.llm.base import LLMConfig, LLMMetadata, LLMProvider
from brok.prompts import PromptTemplate, XMLPromptTemplate, get_prompt_template

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from brok.chat import ContextMessage

logger = logging.getLogger(__name__)


class LlamaCppProvider(LLMProvider):
    """LlamaCpp HTTP server provider implementation.

    Provides integration with llama.cpp's HTTP server for text generation.
    Supports both streaming and complete responses via async generator.

    Compatible with llama.cpp server running with --api-like-OAI or
    the basic HTTP server mode.

    Example:
        >>> provider = LlamaCppProvider(
        ...     base_url="http://localhost:8080",
        ...     model="llama3.2:3b",
        ...     config=LLMConfig(max_tokens=150)
        ... )
        >>> async for chunk in provider.generate("Hello"):
        ...     print(chunk, end="")
        Hello! How can I help you today?
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        config: LLMConfig,
        prompt_template: PromptTemplate | None = None,
        session: aiohttp.ClientSession | None = None,
    ):
        """Initialize LlamaCpp provider.

        Args:
            base_url: LlamaCpp server URL (e.g. "http://localhost:8080")
            model: Model name (e.g. "llama3.2:3b") - informational only
            config: LLM configuration settings
            prompt_template: Optional prompt template (defaults to concise style)
            session: Optional aiohttp session (will create if None)
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.config = config
        self.prompt_template = prompt_template or get_prompt_template("concise")
        self._session = session
        self._session_lock = asyncio.Lock()
        self._last_metadata: LLMMetadata = {}

    async def generate(  # type: ignore[override]
        self,
        prompt: str,
        context: str | None = None,
        context_messages: list[ContextMessage] | None = None,
    ) -> AsyncGenerator[str]:
        """Generate response from LlamaCpp HTTP server.

        Creates a simple prompt with optional context and streams the response.
        Uses the /completion endpoint for text generation.

        Args:
            prompt: The user's input message
            context: Optional conversation context (recent chat history)

        Yields:
            str: Text chunks of the generated response

        Raises:
            LLMConnectionError: When unable to connect to LlamaCpp server
            LLMTimeoutError: When the request times out
            LLMGenerationError: When LlamaCpp returns an error
        """
        # Ensure session is created safely with lock to prevent race conditions
        # Only create new session if we don't have one or if it's a real aiohttp session that's closed
        if not self._session or (
            isinstance(self._session, aiohttp.ClientSession) and self._session.closed
        ):
            async with self._session_lock:
                if not self._session or (
                    isinstance(self._session, aiohttp.ClientSession)
                    and self._session.closed
                ):
                    self._session = aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                    )

        # Build the full prompt with context (KEP-002 Increment B)
        full_prompt = self._build_prompt(prompt, context, context_messages)

        # Prepare request payload for llama.cpp completion endpoint
        payload = {
            "prompt": full_prompt,
            "n_predict": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stop": ["User:", "Assistant:", "\n\n"],  # Common stop sequences
            "stream": False,  # Start with non-streaming for simplicity
        }

        try:
            logger.debug(f"Sending request to LlamaCpp: {self.base_url}/completion")
            async with self._session.post(
                f"{self.base_url}/completion", json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMConnectionError(
                        f"LlamaCpp server returned {response.status}: {error_text}"
                    )

                result = await response.json()

                # Handle LlamaCpp error responses
                if "error" in result:
                    raise LLMGenerationError(f"LlamaCpp error: {result['error']}")

                # Extract the response text
                response_text = result.get("content", "").strip()
                if not response_text:
                    logger.warning("LlamaCpp returned empty response")
                    response_text = "I'm sorry, I couldn't generate a response."

                # Update metadata
                self._last_metadata = {
                    "provider": "llamacpp",
                    "model": self.model,
                    "tokens_used": result.get("tokens_predicted", 0),
                }

                logger.debug(f"Generated response: {len(response_text)} characters")
                yield response_text

        # Handle timeout errors from aiohttp as well as generic asyncio timeouts
        except (TimeoutError, aiohttp.ServerTimeoutError) as e:
            raise LLMTimeoutError(f"LlamaCpp request timed out: {e}") from e
        except aiohttp.ClientError as e:
            raise LLMConnectionError(f"Failed to connect to LlamaCpp: {e}") from e
        except Exception as e:
            if isinstance(e, LLMConnectionError | LLMTimeoutError | LLMGenerationError):
                raise
            raise LLMGenerationError(f"Unexpected error during generation: {e}") from e

    async def health_check(self) -> bool:
        """Check if LlamaCpp server is responding.

        Performs a simple API call to verify connectivity without generating content.
        Uses the /health endpoint if available, or a minimal completion request.

        Returns:
            bool: True if LlamaCpp server is available and responsive

        Raises:
            LLMConnectionError: When unable to connect to LlamaCpp server
        """
        # Create a separate session for health checks with short timeout
        health_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0))

        try:
            # Try the /health endpoint first (if server supports it)
            logger.debug(f"Health check: {self.base_url}/health")
            async with health_session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    logger.debug("Health check result: True (via /health)")
                    return True
        except Exception:
            # /health endpoint might not exist, try a minimal completion instead
            pass

        try:
            # Fallback: minimal completion request
            logger.debug(f"Health check fallback: {self.base_url}/completion")
            payload = {
                "prompt": "test",
                "n_predict": 1,
                "temperature": 0.1,
            }
            async with health_session.post(
                f"{self.base_url}/completion", json=payload
            ) as response:
                is_healthy = bool(response.status == 200)
                logger.debug(f"Health check result: {is_healthy} (via /completion)")
                return is_healthy
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            raise LLMConnectionError(f"LlamaCpp health check failed: {e}") from e
        finally:
            await health_session.close()

    def get_metadata(self) -> LLMMetadata:
        """Get metadata from last generation.

        Returns:
            LLMMetadata: Information about the last generation request
        """
        return self._last_metadata.copy()

    def _build_prompt(
        self,
        prompt: str,
        context: str | None,
        context_messages: list[ContextMessage] | None = None,
    ) -> str:
        """Build the full prompt with optional context and tools using the configured template.

        Args:
            prompt: The user's input message
            context: Optional conversation context (legacy string format)
            context_messages: Optional structured context messages (KEP-002 Increment B)

        Returns:
            str: The complete prompt to send to LlamaCpp
        """
        # Use structured tools with XMLPromptTemplate (KEP-002 Increment C)
        if isinstance(self.prompt_template, XMLPromptTemplate):
            tool_schemas = self.get_tools_schema() if self.has_tools() else None
            return self.prompt_template.build_prompt(
                prompt,
                context,
                tools_description=None,  # Use structured tools instead
                xml_formatting=True,
                context_messages=context_messages,
                tool_schemas=tool_schemas,
            )
        else:
            # Legacy format for non-XML templates
            tools_description = (
                self.get_tools_description() if self.has_tools() else None
            )
            return self.prompt_template.build_prompt(prompt, context, tools_description)

    async def close(self) -> None:
        """Close the HTTP session if we own it."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("Closed LlamaCpp HTTP session")
