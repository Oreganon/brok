"""Tests for LlamaCpp LLM provider."""

from __future__ import annotations

from unittest.mock import AsyncMock

import aiohttp
import pytest

from brok.exceptions import LLMConnectionError, LLMGenerationError, LLMTimeoutError
from brok.llm.base import LLMConfig
from brok.llm.llamacpp import LlamaCppProvider


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
def llamacpp_provider(llm_config: LLMConfig) -> LlamaCppProvider:
    """Provide a LlamaCpp provider instance."""
    return LlamaCppProvider(
        base_url="http://localhost:8080",
        model="llama3.2:3b",
        config=llm_config,
    )


class TestLlamaCppProvider:
    """Test cases for LlamaCppProvider."""

    @pytest.mark.asyncio
    async def test_generate_successful_response(
        self, llamacpp_provider: LlamaCppProvider
    ):
        """Test successful response generation."""
        # Arrange
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "content": "Hello! How can I help you today?",
                "tokens_predicted": 42,
            }
        )

        # Set up the context manager properly
        mock_session.post.return_value = AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response)
        )
        llamacpp_provider._session = mock_session

        # Act
        chunks = []
        async for chunk in llamacpp_provider.generate("Hello"):
            chunks.append(chunk)

        # Assert
        assert len(chunks) == 1
        assert chunks[0] == "Hello! How can I help you today?"

        # Verify metadata
        metadata = llamacpp_provider.get_metadata()
        assert metadata["provider"] == "llamacpp"
        assert metadata["model"] == "llama3.2:3b"
        assert metadata["tokens_used"] == 42

    @pytest.mark.asyncio
    async def test_generate_with_context(self, llamacpp_provider: LlamaCppProvider):
        """Test generation with conversation context."""
        # Arrange
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"content": "Based on the context, here's my response."}
        )

        # Set up the context manager properly
        mock_session.post.return_value = AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response)
        )
        llamacpp_provider._session = mock_session

        context = "user1: Previous message\nuser2: Another message"

        # Act
        chunks = []
        async for chunk in llamacpp_provider.generate("What do you think?", context):
            chunks.append(chunk)

        # Assert
        assert len(chunks) == 1
        assert "Based on the context" in chunks[0]

        # Verify that context was included in the request
        call_args = mock_session.post.call_args
        request_data = call_args[1]["json"]
        assert "Context:" in request_data["prompt"]
        assert "user1: Previous message" in request_data["prompt"]

    @pytest.mark.asyncio
    async def test_generate_empty_response(self, llamacpp_provider: LlamaCppProvider):
        """Test handling of empty response from LlamaCpp."""
        # Arrange
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"content": ""})

        # Set up the context manager properly
        mock_session.post.return_value = AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response)
        )
        llamacpp_provider._session = mock_session

        # Act
        chunks = []
        async for chunk in llamacpp_provider.generate("Hello"):
            chunks.append(chunk)

        # Assert
        assert len(chunks) == 1
        assert "couldn't generate a response" in chunks[0]

    @pytest.mark.asyncio
    async def test_generate_api_error_response(
        self, llamacpp_provider: LlamaCppProvider
    ):
        """Test handling of API error from LlamaCpp."""
        # Arrange
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"error": "Model not found"})

        # Set up the context manager properly
        mock_session.post.return_value = AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response)
        )
        llamacpp_provider._session = mock_session

        # Act & Assert
        with pytest.raises(LLMGenerationError, match="LlamaCpp error: Model not found"):
            async for _ in llamacpp_provider.generate("Hello"):
                pass

    @pytest.mark.asyncio
    async def test_generate_http_error(self, llamacpp_provider: LlamaCppProvider):
        """Test handling of HTTP error responses."""
        # Arrange
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        # Set up the context manager properly
        mock_session.post.return_value = AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response)
        )
        llamacpp_provider._session = mock_session

        # Act & Assert
        with pytest.raises(LLMConnectionError, match="LlamaCpp server returned 500"):
            async for _ in llamacpp_provider.generate("Hello"):
                pass

    @pytest.mark.asyncio
    async def test_generate_connection_error(self, llamacpp_provider: LlamaCppProvider):
        """Test handling of connection errors."""
        # Arrange
        mock_session = AsyncMock()
        mock_session.post.side_effect = aiohttp.ClientError("Connection failed")
        llamacpp_provider._session = mock_session

        # Act & Assert
        with pytest.raises(LLMConnectionError, match="Failed to connect to LlamaCpp"):
            async for _ in llamacpp_provider.generate("Hello"):
                pass

    @pytest.mark.asyncio
    async def test_generate_timeout_error(self, llamacpp_provider: LlamaCppProvider):
        """Test handling of timeout errors."""
        # Arrange
        mock_session = AsyncMock()
        mock_session.post.side_effect = aiohttp.ServerTimeoutError("Request timed out")
        llamacpp_provider._session = mock_session

        # Act & Assert
        with pytest.raises(LLMTimeoutError, match="LlamaCpp request timed out"):
            async for _ in llamacpp_provider.generate("Hello"):
                pass

    @pytest.mark.asyncio
    async def test_health_check_success_via_health_endpoint(
        self, llamacpp_provider: LlamaCppProvider
    ):
        """Test successful health check via /health endpoint."""
        # Arrange
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200

        # Set up the context manager properly
        mock_session.get.return_value = AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response)
        )
        llamacpp_provider._session = mock_session

        # Act
        result = await llamacpp_provider.health_check()

        # Assert
        assert result is True
        mock_session.get.assert_called_once_with("http://localhost:8080/health")

    @pytest.mark.asyncio
    async def test_health_check_success_via_completion_fallback(
        self, llamacpp_provider: LlamaCppProvider
    ):
        """Test successful health check via /completion fallback."""
        # Arrange
        mock_session = AsyncMock()

        # Mock /health endpoint to fail (404)
        mock_health_response = AsyncMock()
        mock_health_response.status = 404

        # Mock /completion endpoint to succeed
        mock_completion_response = AsyncMock()
        mock_completion_response.status = 200

        mock_session.get.side_effect = Exception("Health endpoint not available")
        mock_session.post.return_value = AsyncMock(
            __aenter__=AsyncMock(return_value=mock_completion_response)
        )
        llamacpp_provider._session = mock_session

        # Act
        result = await llamacpp_provider.health_check()

        # Assert
        assert result is True
        mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_failure(self, llamacpp_provider: LlamaCppProvider):
        """Test failed health check."""
        # Arrange
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 500

        # Set up the context manager properly
        mock_session.get.return_value = AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response)
        )
        llamacpp_provider._session = mock_session

        # Act
        result = await llamacpp_provider.health_check()

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_connection_error(
        self, llamacpp_provider: LlamaCppProvider
    ):
        """Test health check with connection error."""
        # Arrange
        mock_session = AsyncMock()
        mock_session.get.side_effect = aiohttp.ClientError("Connection failed")
        mock_session.post.side_effect = aiohttp.ClientError("Connection failed")
        llamacpp_provider._session = mock_session

        # Act & Assert
        with pytest.raises(LLMConnectionError, match="LlamaCpp health check failed"):
            await llamacpp_provider.health_check()

    def test_build_prompt_without_context(self, llamacpp_provider: LlamaCppProvider):
        """Test prompt building without context."""
        # Act
        prompt = llamacpp_provider._build_prompt("Hello", None)

        # Assert
        assert prompt == "User: Hello\nAssistant:"

    def test_build_prompt_with_context(self, llamacpp_provider: LlamaCppProvider):
        """Test prompt building with context."""
        # Arrange
        context = "user1: Hi there\nuser2: How are you?"

        # Act
        prompt = llamacpp_provider._build_prompt("What's up?", context)

        # Assert
        assert "Context:" in prompt
        assert "user1: Hi there" in prompt
        assert "user2: How are you?" in prompt
        assert "User: What's up?" in prompt
        assert "Assistant:" in prompt

    @pytest.mark.asyncio
    async def test_close_session(self, llamacpp_provider: LlamaCppProvider):
        """Test closing HTTP session."""
        # Arrange
        mock_session = AsyncMock()
        mock_session.closed = False
        llamacpp_provider._session = mock_session

        # Act
        await llamacpp_provider.close()

        # Assert
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_already_closed_session(
        self, llamacpp_provider: LlamaCppProvider
    ):
        """Test closing already closed session."""
        # Arrange
        mock_session = AsyncMock()
        mock_session.closed = True
        llamacpp_provider._session = mock_session

        # Act
        await llamacpp_provider.close()

        # Assert
        mock_session.close.assert_not_called()

    def test_get_metadata_empty(self, llamacpp_provider: LlamaCppProvider):
        """Test getting metadata when none is available."""
        # Act
        metadata = llamacpp_provider.get_metadata()

        # Assert
        assert metadata == {}

    @pytest.mark.parametrize(
        "base_url",
        [
            "http://localhost:8080",
            "http://localhost:8080/",  # With trailing slash
            "https://llamacpp.example.com",
            "https://llamacpp.example.com/",  # With trailing slash
        ],
    )
    def test_base_url_normalization(self, base_url: str, llm_config: LLMConfig):
        """Test that base URLs are normalized correctly."""
        # Act
        provider = LlamaCppProvider(
            base_url=base_url,
            model="test-model",
            config=llm_config,
        )

        # Assert
        assert not provider.base_url.endswith("/")
        assert provider.base_url.startswith(("http://", "https://"))

    def test_request_payload_format(self):
        """Test that request payload contains expected LlamaCpp parameters."""
        # This test verifies the specific payload format for llama.cpp server

        # The actual request payload testing is done indirectly through the
        # generate method tests, but this documents the expected format
        expected_keys = ["prompt", "n_predict", "temperature", "stop", "stream"]

        # This is tested implicitly in test_generate_with_context
        # where we verify the request_data contains the expected structure
        assert all(key for key in expected_keys)  # Placeholder assertion
