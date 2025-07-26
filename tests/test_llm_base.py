"""Tests for brok LLM base classes and interfaces."""

from __future__ import annotations

import pytest

from brok.llm.base import LLMConfig, LLMMetadata, LLMProvider


class TestLLMConfig:
    """Test cases for LLMConfig dataclass."""

    def test_llm_config_creation_with_defaults(self):
        """Test creating LLMConfig with default values."""
        # Arrange & Act
        config = LLMConfig(model_name="test-model")

        # Assert
        assert config.model_name == "test-model"
        assert config.max_tokens == 150
        assert config.temperature == 0.7
        assert config.timeout_seconds == 30

    def test_llm_config_creation_with_custom_values(self):
        """Test creating LLMConfig with custom values."""
        # Arrange & Act
        config = LLMConfig(
            model_name="custom-model",
            max_tokens=200,
            temperature=0.9,
            timeout_seconds=60,
        )

        # Assert
        assert config.model_name == "custom-model"
        assert config.max_tokens == 200
        assert config.temperature == 0.9
        assert config.timeout_seconds == 60


class TestLLMMetadata:
    """Test cases for LLMMetadata TypedDict."""

    def test_empty_metadata_creation(self):
        """Test creating empty metadata."""
        # Arrange & Act
        metadata: LLMMetadata = {}

        # Assert
        assert metadata == {}

    def test_partial_metadata_creation(self):
        """Test creating metadata with some fields."""
        # Arrange & Act
        metadata: LLMMetadata = {
            "tokens_used": 42,
            "provider": "test-provider",
        }

        # Assert
        assert metadata["tokens_used"] == 42
        assert metadata["provider"] == "test-provider"
        assert "model" not in metadata

    def test_complete_metadata_creation(self):
        """Test creating metadata with all fields."""
        # Arrange & Act
        metadata: LLMMetadata = {
            "tokens_used": 123,
            "provider": "ollama",
            "model": "llama3.2:3b",
        }

        # Assert
        assert metadata["tokens_used"] == 123
        assert metadata["provider"] == "ollama"
        assert metadata["model"] == "llama3.2:3b"


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, responses: list[str], health_status: bool = True):
        self.responses = responses
        self.health_status = health_status
        self.call_count = 0

    async def generate(self, _prompt: str, _context: str | None = None):
        """Generate mock responses."""
        self.call_count += 1
        for response in self.responses:
            yield response

    async def health_check(self) -> bool:
        """Return mock health status."""
        return self.health_status

    def get_metadata(self) -> LLMMetadata:
        """Return mock metadata."""
        return {
            "tokens_used": len("".join(self.responses)),
            "provider": "mock",
            "model": "test-model",
        }


class TestLLMProvider:
    """Test cases for LLMProvider abstract class."""

    @pytest.mark.asyncio
    async def test_generate_single_response(self):
        """Test generating a single response chunk."""
        # Arrange
        provider = MockLLMProvider(["Hello, world!"])

        # Act
        chunks = []
        async for chunk in provider.generate("test prompt"):
            chunks.append(chunk)

        # Assert
        assert chunks == ["Hello, world!"]
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_multiple_chunks(self):
        """Test generating multiple response chunks."""
        # Arrange
        provider = MockLLMProvider(["Hello, ", "world!"])

        # Act
        chunks = []
        async for chunk in provider.generate("test prompt"):
            chunks.append(chunk)

        # Assert
        assert chunks == ["Hello, ", "world!"]

    @pytest.mark.asyncio
    async def test_generate_with_context(self):
        """Test generating response with context."""
        # Arrange
        provider = MockLLMProvider(["Response with context"])
        context = "Previous conversation history"

        # Act
        chunks = []
        async for chunk in provider.generate("test prompt", context):
            chunks.append(chunk)

        # Assert
        assert chunks == ["Response with context"]

    @pytest.mark.asyncio
    async def test_generate_empty_response(self):
        """Test generating empty response."""
        # Arrange
        provider = MockLLMProvider([])

        # Act
        chunks = []
        async for chunk in provider.generate("test prompt"):
            chunks.append(chunk)

        # Assert
        assert chunks == []

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Test health check when provider is healthy."""
        # Arrange
        provider = MockLLMProvider(["test"], health_status=True)

        # Act
        is_healthy = await provider.health_check()

        # Assert
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        """Test health check when provider is unhealthy."""
        # Arrange
        provider = MockLLMProvider(["test"], health_status=False)

        # Act
        is_healthy = await provider.health_check()

        # Assert
        assert is_healthy is False

    def test_get_metadata_with_custom_implementation(self):
        """Test getting metadata from custom implementation."""
        # Arrange
        provider = MockLLMProvider(["Hello, world!"])

        # Act
        metadata = provider.get_metadata()

        # Assert
        assert metadata["tokens_used"] == 13  # len("Hello, world!")
        assert metadata["provider"] == "mock"
        assert metadata["model"] == "test-model"

    def test_get_metadata_default_implementation(self):
        """Test default metadata implementation returns empty dict."""

        # Arrange
        class MinimalProvider(LLMProvider):
            async def generate(self, _prompt: str, _context: str | None = None):
                yield "test"

            async def health_check(self) -> bool:
                return True

        provider = MinimalProvider()

        # Act
        metadata = provider.get_metadata()

        # Assert
        assert metadata == {}

    @pytest.mark.asyncio
    async def test_join_generated_chunks(self):
        """Test joining all generated chunks into complete response."""
        # Arrange
        provider = MockLLMProvider(["Hello, ", "how ", "are ", "you?"])

        # Act
        full_response = "".join([chunk async for chunk in provider.generate("test")])

        # Assert
        assert full_response == "Hello, how are you?"
