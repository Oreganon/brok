"""Ollama LLM provider implementation."""

from __future__ import annotations

import asyncio  # Added to catch asyncio.TimeoutError
import logging
from typing import TYPE_CHECKING

import aiohttp

from brok.exceptions import LLMConnectionError, LLMGenerationError, LLMTimeoutError
from brok.llm.base import LLMConfig, LLMMetadata, LLMProvider
from brok.prompts import PromptTemplate, get_prompt_template

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama API provider implementation.

    Provides integration with Ollama's REST API for text generation.
    Supports both streaming and complete responses via async generator.

    Example:
        >>> provider = OllamaProvider(
        ...     base_url="http://localhost:11434",
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
        """Initialize Ollama provider.

        Args:
            base_url: Ollama server URL (e.g. "http://localhost:11434")
            model: Model name (e.g. "llama3.2:3b")
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

    async def generate(
        self, prompt: str, context: str | None = None
    ) -> AsyncGenerator[str]:
        """Generate response from Ollama API.

        Creates a simple prompt with optional context and streams the response.
        For Phase 2, we use non-streaming mode for simplicity.

        Args:
            prompt: The user's input message
            context: Optional conversation context (recent chat history)

        Yields:
            str: Text chunks of the generated response

        Raises:
            LLMConnectionError: When unable to connect to Ollama
            LLMTimeoutError: When the request times out
            LLMGenerationError: When Ollama returns an error
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

        # Build the full prompt with context
        full_prompt = self._build_prompt(prompt, context)

        # Prepare request payload
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,  # Start with non-streaming for simplicity
            "options": {
                "num_predict": self.config.max_tokens,
                "temperature": self.config.temperature,
            },
        }

        try:
            logger.debug(f"Sending request to Ollama: {self.base_url}/api/generate")
            async with self._session.post(
                f"{self.base_url}/api/generate", json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMConnectionError(
                        f"Ollama API returned {response.status}: {error_text}"
                    )

                result = await response.json()

                # Handle Ollama error responses
                if "error" in result:
                    raise LLMGenerationError(f"Ollama error: {result['error']}")

                # Extract the response text
                response_text = result.get("response", "").strip()
                if not response_text:
                    logger.warning("Ollama returned empty response")
                    response_text = "I'm sorry, I couldn't generate a response."

                # Update metadata
                self._last_metadata = {
                    "provider": "ollama",
                    "model": self.model,
                    "tokens_used": result.get("eval_count", 0),
                }

                logger.debug(f"Generated response: {len(response_text)} characters")
                yield response_text

        # Handle timeout errors from aiohttp as well as generic asyncio timeouts
        except (TimeoutError, aiohttp.ServerTimeoutError) as e:
            raise LLMTimeoutError(f"Ollama request timed out: {e}") from e
        except aiohttp.ClientError as e:
            raise LLMConnectionError(f"Failed to connect to Ollama: {e}") from e
        except Exception as e:
            if isinstance(e, LLMConnectionError | LLMTimeoutError | LLMGenerationError):
                raise
            raise LLMGenerationError(f"Unexpected error during generation: {e}") from e

    async def health_check(self) -> bool:
        """Check if Ollama is responding.

        Performs a simple API call to verify connectivity without generating content.

        Returns:
            bool: True if Ollama is available and responsive

        Raises:
            LLMConnectionError: When unable to connect to Ollama
        """
        # Create a separate session for health checks with short timeout
        health_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0))

        try:
            logger.debug(f"Health check: {self.base_url}/api/tags")
            async with health_session.get(f"{self.base_url}/api/tags") as response:
                is_healthy = response.status == 200
                logger.debug(f"Health check result: {is_healthy}")
                return is_healthy
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            raise LLMConnectionError(f"Ollama health check failed: {e}") from e
        finally:
            await health_session.close()

    def get_metadata(self) -> LLMMetadata:
        """Get metadata from last generation.

        Returns:
            LLMMetadata: Information about the last generation request
        """
        return self._last_metadata.copy()

    def _build_prompt(self, prompt: str, context: str | None) -> str:
        """Build the full prompt with optional context and tools using the configured template.

        Args:
            prompt: The user's input message
            context: Optional conversation context

        Returns:
            str: The complete prompt to send to Ollama
        """
        tools_description = self.get_tools_description() if self.has_tools() else None
        return self.prompt_template.build_prompt(prompt, context, tools_description)

    async def close(self) -> None:
        """Close the HTTP session if we own it."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("Closed Ollama HTTP session")
