"""Accurate token counting for prompt optimization (KEP-002 Increment D).

This module provides precise token measurement using tiktoken, specifically
optimized for smaller models (2B parameters) where token efficiency is critical.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from brok.chat import ContextMessage

    # Define encoding protocol for type checking
    class EncodingProtocol(Protocol):
        """Protocol for tiktoken Encoding for type checking."""

        def encode(self, text: str) -> list[int]: ...


# Handle tiktoken import
try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None  # type: ignore[assignment]
    _TIKTOKEN_AVAILABLE = False


@dataclass
class TokenMeasurement:
    """Results from token counting analysis."""

    token_count: int
    character_count: int
    measurement_time_ms: float
    encoding_used: str
    efficiency_ratio: float  # tokens per character

    @property
    def is_efficient(self) -> bool:
        """Check if the token efficiency is reasonable for smaller models."""
        # For 2B models, we want high density (< 0.4 tokens per char is good)
        return self.efficiency_ratio < 0.4


class TokenCounter:
    """Accurate token counter optimized for 2B parameter models.

    Provides precise token measurement using tiktoken with fallback to
    estimation for environments where tiktoken is unavailable.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo") -> None:
        """Initialize token counter.

        Args:
            model_name: Model name for tiktoken encoding selection.
                       Defaults to gpt-3.5-turbo which works well for most models.
        """
        self._model_name = model_name
        self._encoding = self._get_encoding()
        self._fallback_ratio = 0.25  # Conservative estimate: 1 token per 4 chars

    def _get_encoding(self) -> EncodingProtocol | None:
        """Get tiktoken encoding, with graceful fallback."""
        if not _TIKTOKEN_AVAILABLE or tiktoken is None:
            return None

        try:
            # Try model-specific encoding first
            return tiktoken.encoding_for_model(self._model_name)
        except KeyError:
            try:
                # Fall back to cl100k_base (GPT-3.5/4 encoding)
                return tiktoken.get_encoding("cl100k_base")
            except Exception:
                return None

    def count_tokens(self, text: str) -> TokenMeasurement:
        """Count tokens in text with performance measurement.

        Args:
            text: Text to analyze

        Returns:
            TokenMeasurement: Detailed token analysis
        """
        start_time = time.perf_counter()
        char_count = len(text)

        if self._encoding is not None:
            # Use accurate tiktoken counting
            try:
                tokens = self._encoding.encode(text)
                token_count = len(tokens)
                encoding_used = "tiktoken"
            except Exception:
                # Fallback if tiktoken fails
                token_count = max(1, int(char_count * self._fallback_ratio))
                encoding_used = "fallback"
        else:
            # Use conservative estimation
            token_count = max(1, int(char_count * self._fallback_ratio))
            encoding_used = "estimation"

        measurement_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        efficiency_ratio = token_count / max(1, char_count)

        return TokenMeasurement(
            token_count=token_count,
            character_count=char_count,
            measurement_time_ms=measurement_time,
            encoding_used=encoding_used,
            efficiency_ratio=efficiency_ratio,
        )

    def measure_prompt_overhead(
        self, xml_prompt: str, text_prompt: str
    ) -> dict[str, float]:
        """Measure token overhead of XML vs text prompts.

        Args:
            xml_prompt: XML-formatted prompt
            text_prompt: Plain text equivalent prompt

        Returns:
            dict: Overhead analysis with percentages and absolute differences
        """
        xml_measurement = self.count_tokens(xml_prompt)
        text_measurement = self.count_tokens(text_prompt)

        token_overhead = xml_measurement.token_count - text_measurement.token_count
        char_overhead = (
            xml_measurement.character_count - text_measurement.character_count
        )

        # Calculate percentage overhead
        token_overhead_pct = (
            token_overhead / max(1, text_measurement.token_count)
        ) * 100
        char_overhead_pct = (
            char_overhead / max(1, text_measurement.character_count)
        ) * 100

        return {
            "xml_tokens": xml_measurement.token_count,
            "text_tokens": text_measurement.token_count,
            "token_overhead_absolute": token_overhead,
            "token_overhead_percent": token_overhead_pct,
            "xml_chars": xml_measurement.character_count,
            "text_chars": text_measurement.character_count,
            "char_overhead_absolute": char_overhead,
            "char_overhead_percent": char_overhead_pct,
            "xml_efficiency": xml_measurement.efficiency_ratio,
            "text_efficiency": text_measurement.efficiency_ratio,
            "meets_target": token_overhead_pct < 20.0,  # KEP-002 target: <20%
        }

    def optimize_context_for_budget(
        self,
        context_messages: list[ContextMessage],
        token_budget: int,
        preserve_recent: int = 3,
    ) -> list[ContextMessage]:
        """Optimize context messages to fit within token budget.

        Specifically designed for smaller 2B models with limited context windows.

        Args:
            context_messages: Messages to optimize
            token_budget: Maximum tokens allowed for context
            preserve_recent: Always preserve this many recent messages

        Returns:
            list[ContextMessage]: Optimized message list within budget
        """
        if not context_messages:
            return []

        # Always preserve most recent messages
        preserved = context_messages[-preserve_recent:] if preserve_recent > 0 else []
        candidates = (
            context_messages[:-preserve_recent]
            if preserve_recent > 0
            else context_messages
        )

        # Calculate tokens for preserved messages
        preserved_tokens = 0
        for msg in preserved:
            content_tokens = self.count_tokens(msg.content).token_count
            # Add overhead for sender name and formatting (conservative estimate)
            preserved_tokens += content_tokens + len(msg.sender) // 4 + 5

        remaining_budget = max(0, token_budget - preserved_tokens)

        # Select additional messages that fit in remaining budget
        selected = []
        current_tokens = 0

        # Process candidates in reverse order (newest first after preserved)
        for msg in reversed(candidates):
            content_tokens = self.count_tokens(msg.content).token_count
            msg_tokens = content_tokens + len(msg.sender) // 4 + 5

            if current_tokens + msg_tokens <= remaining_budget:
                selected.append(msg)
                current_tokens += msg_tokens
            else:
                break

        # Combine selected messages (restore chronological order) with preserved
        return list(reversed(selected)) + preserved

    def validate_performance(self, text: str, max_time_ms: float = 5.0) -> bool:
        """Validate that token counting meets performance requirements.

        KEP-002 target: <5ms overhead for token counting.

        Args:
            text: Sample text to measure
            max_time_ms: Maximum allowed time in milliseconds

        Returns:
            bool: True if performance meets requirements
        """
        measurement = self.count_tokens(text)
        return measurement.measurement_time_ms < max_time_ms


# Global instance for efficient reuse
_default_counter: TokenCounter | None = None


class TokenCounterRegistry:
    """Registry for token counter instances to avoid global variables."""

    _instance: TokenCounterRegistry | None = None
    _counters: dict[str, TokenCounter]

    def __new__(cls) -> TokenCounterRegistry:
        """Singleton pattern for registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._counters = {}
        return cls._instance

    def get_counter(self, model_name: str = "gpt-3.5-turbo") -> TokenCounter:
        """Get or create a token counter instance.

        Args:
            model_name: Model name for encoding selection

        Returns:
            TokenCounter: Reusable token counter instance
        """
        if model_name not in self._counters:
            self._counters[model_name] = TokenCounter(model_name)
        return self._counters[model_name]


def get_token_counter(model_name: str = "gpt-3.5-turbo") -> TokenCounter:
    """Get or create a token counter instance.

    Args:
        model_name: Model name for encoding selection

    Returns:
        TokenCounter: Reusable token counter instance
    """
    registry = TokenCounterRegistry()
    return registry.get_counter(model_name)


def count_tokens_fast(text: str) -> int:
    """Quick token count for common use cases.

    Args:
        text: Text to count tokens for

    Returns:
        int: Estimated token count
    """
    return get_token_counter().count_tokens(text).token_count
