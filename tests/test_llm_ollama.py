"""Tests for Ollama LLM provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from brok.exceptions import LLMConnectionError, LLMGenerationError, LLMTimeoutError
from brok.llm.base import LLMConfig
from brok.llm.ollama import OllamaProvider


class AsyncContextManagerMock:
    """Helper class to create proper async context manager mocks for aiohttp responses."""

    def __init__(self, mock_response):
        self.mock_response = mock_response

    async def __aenter__(self):
        return self.mock_response

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


@pytest.fixture
def llm_config() -> LLMConfig:
    """Provide a sample LLM configuration."""
    return LLMConfig(
        model_name="llama3.2:3b",
        max_tokens=150,
        temperature=0.7,
        timeout_seconds=30,
    )


@pytest.fixture
def ollama_provider(llm_config: LLMConfig) -> OllamaProvider:
    """Provide an Ollama provider instance."""
    return OllamaProvider(
        base_url="http://localhost:11434",
        model="llama3.2:3b",
        config=llm_config,
    )


class TestOllamaProvider:
    """Test cases for OllamaProvider."""

    @pytest.mark.asyncio
    async def test_generate_successful_response(self, ollama_provider: OllamaProvider):
        """Test successful response generation."""
        # Arrange
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.closed = False  # Make it appear as not closed
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "response": "Hello! How can I help you today?",
                "eval_count": 42,
            }
        )

        # Set up the context manager properly
        mock_session.post.return_value = AsyncContextManagerMock(mock_response)
        ollama_provider._session = mock_session

        # Act
        chunks = []
        async for chunk in ollama_provider.generate("Hello"):
            chunks.append(chunk)

        # Assert
        assert len(chunks) == 1
        assert chunks[0] == "Hello! How can I help you today?"

        # Verify metadata
        metadata = ollama_provider.get_metadata()
        assert metadata["provider"] == "ollama"
        assert metadata["model"] == "llama3.2:3b"
        assert metadata["tokens_used"] == 42

    @pytest.mark.asyncio
    async def test_generate_with_context(self, ollama_provider: OllamaProvider):
        """Test generation with conversation context."""
        # Arrange
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.closed = False  # Make it appear as not closed
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"response": "Based on the context, here's my response."}
        )

        # Set up the context manager properly
        mock_session.post.return_value = AsyncContextManagerMock(mock_response)
        ollama_provider._session = mock_session

        context = "user1: Previous message\nuser2: Another message"

        # Act
        chunks = []
        async for chunk in ollama_provider.generate("What do you think?", context):
            chunks.append(chunk)

        # Assert
        assert len(chunks) == 1
        assert "Based on the context" in chunks[0]

        # Verify that context was included in the request
        call_args = mock_session.post.call_args
        request_data = call_args[1]["json"]
        assert "## Recent Context" in request_data["prompt"]
        assert "user1: Previous message" in request_data["prompt"]

    @pytest.mark.asyncio
    async def test_generate_empty_response(self, ollama_provider: OllamaProvider):
        """Test handling of empty response from Ollama."""
        # Arrange
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.closed = False  # Make it appear as not closed
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"response": ""})

        # Set up the context manager properly
        mock_session.post.return_value = AsyncContextManagerMock(mock_response)
        ollama_provider._session = mock_session

        # Act
        chunks = []
        async for chunk in ollama_provider.generate("Hello"):
            chunks.append(chunk)

        # Assert
        assert len(chunks) == 1
        assert "couldn't generate a response" in chunks[0]

    @pytest.mark.asyncio
    async def test_generate_api_error_response(self, ollama_provider: OllamaProvider):
        """Test handling of API error from Ollama."""
        # Arrange
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.closed = False  # Make it appear as not closed
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"error": "Model not found"})

        # Set up the context manager properly
        mock_session.post.return_value = AsyncContextManagerMock(mock_response)
        ollama_provider._session = mock_session

        # Act & Assert
        with pytest.raises(LLMGenerationError, match="Ollama error: Model not found"):
            async for _ in ollama_provider.generate("Hello"):
                pass

    @pytest.mark.asyncio
    async def test_generate_http_error(self, ollama_provider: OllamaProvider):
        """Test handling of HTTP error responses."""
        # Arrange
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.closed = False  # Make it appear as not closed
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        # Set up the context manager properly
        mock_session.post.return_value = AsyncContextManagerMock(mock_response)
        ollama_provider._session = mock_session

        # Act & Assert
        with pytest.raises(LLMConnectionError, match="Ollama API returned 500"):
            async for _ in ollama_provider.generate("Hello"):
                pass

    @pytest.mark.asyncio
    async def test_generate_connection_error(self, ollama_provider: OllamaProvider):
        """Test handling of connection errors."""
        # Arrange
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.closed = False  # Make it appear as not closed
        mock_session.post.side_effect = aiohttp.ClientError("Connection failed")
        ollama_provider._session = mock_session

        # Act & Assert
        with pytest.raises(LLMConnectionError, match="Failed to connect to Ollama"):
            async for _ in ollama_provider.generate("Hello"):
                pass

    @pytest.mark.asyncio
    async def test_generate_timeout_error(self, ollama_provider: OllamaProvider):
        """Test handling of timeout errors."""
        # Arrange
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.closed = False  # Make it appear as not closed
        mock_session.post.side_effect = aiohttp.ServerTimeoutError("Request timed out")
        ollama_provider._session = mock_session

        # Act & Assert
        with pytest.raises(LLMTimeoutError, match="Ollama request timed out"):
            async for _ in ollama_provider.generate("Hello"):
                pass

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_health_check_success(
        self, mock_session_class, ollama_provider: OllamaProvider
    ):
        """Test successful health check."""
        # Arrange
        mock_session = Mock()
        mock_session.closed = False  # Make it appear as not closed
        mock_session_class.return_value = mock_session

        mock_response = AsyncMock()
        mock_response.status = 200

        # Set up the context manager properly
        mock_session.get = Mock(return_value=AsyncContextManagerMock(mock_response))
        mock_session.close = AsyncMock()  # Add async close method

        # Act
        result = await ollama_provider.health_check()

        # Assert
        assert result is True
        mock_session.get.assert_called_once_with("http://localhost:11434/api/tags")

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_health_check_failure(
        self, mock_session_class, ollama_provider: OllamaProvider
    ):
        """Test failed health check."""
        # Arrange
        mock_session = AsyncMock()
        mock_session.closed = False  # Make it appear as not closed
        mock_session_class.return_value = mock_session
        mock_response = AsyncMock()
        mock_response.status = 500

        # Set up the context manager properly
        mock_session.get.return_value = AsyncContextManagerMock(mock_response)

        # Act & Assert
        with pytest.raises(LLMConnectionError, match="Ollama health check failed"):
            await ollama_provider.health_check()

    @pytest.mark.asyncio
    async def test_health_check_connection_error(self, ollama_provider: OllamaProvider):
        """Test health check with connection error."""
        # Arrange
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.closed = False  # Make it appear as not closed
        mock_session.get.side_effect = aiohttp.ClientError("Connection failed")
        ollama_provider._session = mock_session

        # Act & Assert
        with pytest.raises(LLMConnectionError, match="Ollama health check failed"):
            await ollama_provider.health_check()

    def test_build_prompt_without_context(self, ollama_provider: OllamaProvider):
        """Test prompt building without context."""
        # Act
        prompt = ollama_provider._build_prompt("Hello", None)

        # Assert
        assert "## Request" in prompt
        assert "Hello" in prompt
        assert "Assistant:" in prompt
        assert "helpful AI assistant" in prompt  # Verify system prompt is included
        assert "chat room" in prompt  # Verify context is included

    def test_build_prompt_with_context(self, ollama_provider: OllamaProvider):
        """Test prompt building with context."""
        # Arrange
        context = "user1: Hi there\nuser2: How are you?"

        # Act
        prompt = ollama_provider._build_prompt("What's up?", context)

        # Assert
        assert "## Recent Context" in prompt
        assert "user1: Hi there" in prompt
        assert "user2: How are you?" in prompt
        assert "## Request" in prompt
        assert "What's up?" in prompt
        assert "Assistant:" in prompt

    @pytest.mark.asyncio
    async def test_close_session(self, ollama_provider: OllamaProvider):
        """Test closing HTTP session."""
        # Arrange
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.closed = False  # Make it appear as not closed
        mock_session.closed = False
        ollama_provider._session = mock_session

        # Act
        await ollama_provider.close()

        # Assert
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_already_closed_session(self, ollama_provider: OllamaProvider):
        """Test closing already closed session."""
        # Arrange
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.closed = False  # Make it appear as not closed
        mock_session.closed = True
        ollama_provider._session = mock_session

        # Act
        await ollama_provider.close()

        # Assert
        mock_session.close.assert_not_called()

    def test_get_metadata_empty(self, ollama_provider: OllamaProvider):
        """Test getting metadata when none is available."""
        # Act
        metadata = ollama_provider.get_metadata()

        # Assert
        assert metadata == {}

    @pytest.mark.parametrize(
        "base_url",
        [
            "http://localhost:11434",
            "http://localhost:11434/",  # With trailing slash
            "https://ollama.example.com",
            "https://ollama.example.com/",  # With trailing slash
        ],
    )
    def test_base_url_normalization(self, base_url: str, llm_config: LLMConfig):
        """Test that base URLs are normalized correctly."""
        # Act
        provider = OllamaProvider(
            base_url=base_url,
            model="test-model",
            config=llm_config,
        )

        # Assert
        assert not provider.base_url.endswith("/")
        assert provider.base_url.startswith(("http://", "https://"))
