"""Tests for brok custom exceptions."""

from __future__ import annotations

import pytest

from brok.exceptions import (
    BrokError,
    ChatAuthenticationError,
    ChatClientError,
    ChatConnectionError,
    ConfigurationError,
    LLMConnectionError,
    LLMGenerationError,
    LLMProviderError,
    LLMTimeoutError,
)


class TestExceptionHierarchy:
    """Test cases for exception hierarchy and inheritance."""

    def test_brok_error_base_exception(self):
        """Test that BrokError inherits from Exception."""
        # Arrange & Act
        error = BrokError("test message")

        # Assert
        assert isinstance(error, Exception)
        assert str(error) == "test message"

    def test_configuration_error_inheritance(self):
        """Test that ConfigurationError inherits from BrokError."""
        # Arrange & Act
        error = ConfigurationError("config error")

        # Assert
        assert isinstance(error, BrokError)
        assert isinstance(error, Exception)
        assert str(error) == "config error"

    def test_llm_provider_error_inheritance(self):
        """Test that LLMProviderError inherits from BrokError."""
        # Arrange & Act
        error = LLMProviderError("llm error")

        # Assert
        assert isinstance(error, BrokError)
        assert isinstance(error, Exception)
        assert str(error) == "llm error"

    def test_llm_connection_error_inheritance(self):
        """Test that LLMConnectionError inherits from LLMProviderError."""
        # Arrange & Act
        error = LLMConnectionError("connection failed")

        # Assert
        assert isinstance(error, LLMProviderError)
        assert isinstance(error, BrokError)
        assert isinstance(error, Exception)
        assert str(error) == "connection failed"

    def test_llm_timeout_error_inheritance(self):
        """Test that LLMTimeoutError inherits from LLMProviderError."""
        # Arrange & Act
        error = LLMTimeoutError("request timed out")

        # Assert
        assert isinstance(error, LLMProviderError)
        assert isinstance(error, BrokError)
        assert isinstance(error, Exception)
        assert str(error) == "request timed out"

    def test_llm_generation_error_inheritance(self):
        """Test that LLMGenerationError inherits from LLMProviderError."""
        # Arrange & Act
        error = LLMGenerationError("generation failed")

        # Assert
        assert isinstance(error, LLMProviderError)
        assert isinstance(error, BrokError)
        assert isinstance(error, Exception)
        assert str(error) == "generation failed"

    def test_chat_client_error_inheritance(self):
        """Test that ChatClientError inherits from BrokError."""
        # Arrange & Act
        error = ChatClientError("chat error")

        # Assert
        assert isinstance(error, BrokError)
        assert isinstance(error, Exception)
        assert str(error) == "chat error"

    def test_chat_connection_error_inheritance(self):
        """Test that ChatConnectionError inherits from ChatClientError."""
        # Arrange & Act
        error = ChatConnectionError("chat connection failed")

        # Assert
        assert isinstance(error, ChatClientError)
        assert isinstance(error, BrokError)
        assert isinstance(error, Exception)
        assert str(error) == "chat connection failed"

    def test_chat_authentication_error_inheritance(self):
        """Test that ChatAuthenticationError inherits from ChatClientError."""
        # Arrange & Act
        error = ChatAuthenticationError("auth failed")

        # Assert
        assert isinstance(error, ChatClientError)
        assert isinstance(error, BrokError)
        assert isinstance(error, Exception)
        assert str(error) == "auth failed"


class TestExceptionCatching:
    """Test cases for exception catching patterns."""

    def test_catch_specific_llm_errors(self):
        """Test catching specific LLM errors."""
        # Test LLMConnectionError
        with pytest.raises(LLMConnectionError):
            raise LLMConnectionError("connection failed")

        # Test LLMTimeoutError
        with pytest.raises(LLMTimeoutError):
            raise LLMTimeoutError("timeout")

        # Test LLMGenerationError
        with pytest.raises(LLMGenerationError):
            raise LLMGenerationError("generation failed")

    def test_catch_llm_provider_error_base(self):
        """Test catching any LLM provider error via base class."""
        # All LLM-specific errors should be catchable as LLMProviderError
        with pytest.raises(LLMProviderError):
            raise LLMConnectionError("connection failed")

        with pytest.raises(LLMProviderError):
            raise LLMTimeoutError("timeout")

        with pytest.raises(LLMProviderError):
            raise LLMGenerationError("generation failed")

    def test_catch_chat_client_error_base(self):
        """Test catching any chat client error via base class."""
        # All chat-specific errors should be catchable as ChatClientError
        with pytest.raises(ChatClientError):
            raise ChatConnectionError("connection failed")

        with pytest.raises(ChatClientError):
            raise ChatAuthenticationError("auth failed")

    def test_catch_any_brok_error(self):
        """Test catching any brok error via base class."""
        # All brok errors should be catchable as BrokError
        with pytest.raises(BrokError):
            raise ConfigurationError("config error")

        with pytest.raises(BrokError):
            raise LLMConnectionError("llm error")

        with pytest.raises(BrokError):
            raise ChatConnectionError("chat error")

    def test_exception_chaining_with_cause(self):
        """Test exception chaining with original cause."""
        # Arrange
        original_error = ValueError("original error")

        # Act & Assert
        with pytest.raises(ConfigurationError) as exc_info:
            try:
                raise original_error
            except ValueError as e:
                raise ConfigurationError("wrapped error") from e

        assert exc_info.value.__cause__ is original_error
        assert str(exc_info.value) == "wrapped error"
