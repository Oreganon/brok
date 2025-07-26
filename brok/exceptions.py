"""Custom exceptions for the brok chatbot."""

from __future__ import annotations


class BrokError(Exception):
    """Base exception for all brok chatbot errors."""


class ConfigurationError(BrokError):
    """Raised when configuration is invalid or missing required values."""


class LLMProviderError(BrokError):
    """Base exception for LLM provider related errors."""


class LLMConnectionError(LLMProviderError):
    """Raised when unable to connect to LLM provider."""


class LLMTimeoutError(LLMProviderError):
    """Raised when LLM provider request times out."""


class LLMGenerationError(LLMProviderError):
    """Raised when LLM fails to generate a response."""


class ChatClientError(BrokError):
    """Base exception for chat client related errors."""


class ChatConnectionError(ChatClientError):
    """Raised when unable to connect to chat service."""


class ChatAuthenticationError(ChatClientError):
    """Raised when chat authentication fails."""
