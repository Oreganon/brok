"""Chat client wrapper and message processing for strims.gg integration."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import logging
import sys
import time
from typing import TYPE_CHECKING

from wsggpy import AsyncSession, ChatEnvironment, Message, RoomAction

from brok.exceptions import ChatAuthenticationError, ChatConnectionError

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def is_mention(message: str, bot_name: str) -> bool:
    """Check if message mentions the bot.

    Detects various mention patterns:
    - @botname
    - botname:
    - botname,
    - botname (at start of message)

    Args:
        message: The chat message text
        bot_name: The bot's name to check for mentions

    Returns:
        bool: True if the message mentions the bot
    """
    if not message.strip():
        return False

    # Normalize both message and bot name for comparison
    message_lower = message.lower().strip()
    bot_name_lower = bot_name.lower()

    # Pattern 1: @botname (anywhere in message)
    if f"@{bot_name_lower}" in message_lower:
        return True

    # Pattern 2: botname: (at start of message)
    if message_lower.startswith(f"{bot_name_lower}:"):
        return True

    # Pattern 3: botname, (at start of message)
    if message_lower.startswith(f"{bot_name_lower},"):
        return True

    # Pattern 4: botname (as first word)
    words = message_lower.split()
    return bool(words and words[0] == bot_name_lower)


def parse_command(message: str, bot_name: str) -> ParsedCommand | None:  # noqa: PLR0911
    """Parse a command from a chat message.

    Supports various command formats:
    - !command args
    - @botname command args
    - botname: command args
    - botname, command args
    - botname command args

    Args:
        message: The chat message text
        bot_name: The bot's name

    Returns:
        ParsedCommand | None: Parsed command or None if not a command
    """
    if not message.strip():
        return None

    message = message.strip()
    message_lower = message.lower()
    bot_name_lower = bot_name.lower()

    # Pattern 1: !command args
    if message.startswith("!"):
        parts = message[1:].split()
        if parts:
            return ParsedCommand(
                command=parts[0].lower(), args=parts[1:], full_text=message
            )

    # Pattern 2: @botname command args
    at_mention_pattern = f"@{bot_name_lower}"
    if message_lower.startswith(at_mention_pattern):
        remaining = message[len(at_mention_pattern) :].strip()
        parts = remaining.split()
        if parts:
            return ParsedCommand(
                command=parts[0].lower(), args=parts[1:], full_text=message
            )

    # Pattern 3: botname: command args
    colon_pattern = f"{bot_name_lower}:"
    if message_lower.startswith(colon_pattern):
        remaining = message[len(colon_pattern) :].strip()
        parts = remaining.split()
        if parts:
            return ParsedCommand(
                command=parts[0].lower(), args=parts[1:], full_text=message
            )

    # Pattern 4: botname, command args
    comma_pattern = f"{bot_name_lower},"
    if message_lower.startswith(comma_pattern):
        remaining = message[len(comma_pattern) :].strip()
        parts = remaining.split()
        if parts:
            return ParsedCommand(
                command=parts[0].lower(), args=parts[1:], full_text=message
            )

    # Pattern 5: botname command args (botname as first word)
    words = message.split()
    if words and words[0].lower() == bot_name_lower and len(words) > 1:
        return ParsedCommand(
            command=words[1].lower(), args=words[2:], full_text=message
        )

    return None


@dataclass
class ProcessedMessage:
    """Represents a processed chat message ready for LLM processing.

    Attributes:
        original_message: The original chat message text
        sender: Username of the message sender
        timestamp: Unix timestamp when message was received
        context: Optional conversation context (recent chat history)
        message_type: Type of message (keyword, mention, command)
        command: Parsed command name if message_type is "command"
        command_args: Parsed command arguments if message_type is "command"
    """

    original_message: str
    sender: str
    timestamp: float
    context: str | None = None
    message_type: str = "keyword"  # "keyword", "mention", "command"
    command: str | None = None
    command_args: list[str] | None = None


@dataclass
class ParsedCommand:
    """Represents a parsed command from a chat message.

    Attributes:
        command: The command name (e.g., "help", "status")
        args: List of command arguments
        full_text: The original command text
    """

    command: str
    args: list[str]
    full_text: str


@dataclass
class ContextMessage:
    """Simple message with essential metadata for enhanced context management.

    This data structure supports KEP-001 Increment A by providing structured
    message storage while maintaining backward compatibility.

    Attributes:
        content: The message text content
        sender: Username of the message sender
        timestamp: When the message was created
        is_bot: Whether this message was sent by the bot
        message_id: Unique identifier for the message
    """

    content: str
    sender: str
    timestamp: datetime
    is_bot: bool
    message_id: str = field(default_factory=lambda: str(time.time_ns()))


class ChatClient:
    """Strims chat client wrapper with message processing.

    Handles connection to strims.gg chat, message filtering, context management,
    and response sending. Integrates with wsggpy for WebSocket communication.

    Example:
        >>> client = ChatClient(
        ...     response_filters=[lambda msg, sender: msg.startswith("!bot")],
        ...     context_window_size=10
        ... )
        >>> await client.connect(jwt_token=None, environment="production")
        >>> # Client will now process incoming messages
    """

    def __init__(
        self,
        response_filters: list[Callable[[str, str], bool]],
        context_window_size: int = 10,
        bot_name: str = "brok",
        respond_to_mentions: bool = True,
        respond_to_commands: bool = True,
        ignore_users: list[str] | None = None,
        enhanced_context: bool = True,  # Now defaults to True
        max_context_tokens: int = 500,
        prioritize_mentions: bool = True,
        include_bot_responses: bool = True,
    ):
        """Initialize chat client with enhanced context capabilities.

        Args:
            response_filters: List of functions to determine if bot should respond
            context_window_size: Number of recent messages to keep for context
            bot_name: Bot name for mention detection
            respond_to_mentions: Whether to respond to mentions
            respond_to_commands: Whether to parse and respond to commands
            ignore_users: List of usernames to ignore (bot name is automatically added)
            enhanced_context: Feature flag for structured context (KEP-001)
            max_context_tokens: Maximum tokens to include in context
            prioritize_mentions: Whether to prioritize mentions in context
            include_bot_responses: Whether to include bot responses in context
        """
        self._filters = response_filters
        self._context_window_size = context_window_size
        self._bot_name = bot_name
        self._respond_to_mentions = respond_to_mentions
        self._respond_to_commands = respond_to_commands

        # Enhanced context settings (KEP-001)
        self._enhanced_context = enhanced_context
        self._max_context_tokens = max_context_tokens
        self._prioritize_mentions = prioritize_mentions
        self._include_bot_responses = include_bot_responses

        # Initialize ignore_users list and automatically add bot name (all lowercase for case-insensitive matching)
        self._ignore_users = {user.lower() for user in (ignore_users or [])}
        self._ignore_users.add(bot_name.lower())  # Add bot name (case-insensitive)

        # Context storage - Enhanced context by default
        if self._enhanced_context:
            self._context_messages_structured: deque[ContextMessage] = deque(
                maxlen=context_window_size
            )
        else:
            # Legacy fallback for backward compatibility
            self._context_messages_legacy: list[str] = []

        self._processing_queue: asyncio.Queue[ProcessedMessage] = asyncio.Queue()
        self._session: AsyncSession | None = None
        self._is_connected = False

    async def connect(self, jwt_token: str | None, environment: str) -> None:
        """Connect to strims chat.

        Args:
            jwt_token: Optional JWT token for authentication
            environment: "production" or "dev" environment

        Raises:
            ChatConnectionError: When unable to connect
            ChatAuthenticationError: When authentication fails
        """
        try:
            # Determine chat environment
            chat_env = (
                ChatEnvironment.DEV
                if environment == "dev"
                else ChatEnvironment.PRODUCTION
            )

            logger.info(f"Connecting to strims.gg {environment} chat...")

            # Create async session
            self._session = AsyncSession(
                login_key=jwt_token,
                url=chat_env,
            )

            # Add message handlers
            self._session.add_message_handler(self._on_message)
            self._session.add_join_handler(self._on_join)
            self._session.add_quit_handler(self._on_quit)

            # Connect to chat
            await self._session.open()
            self._is_connected = True

            auth_status = "authenticated" if jwt_token else "anonymous"
            logger.info(f"✅ Connected to {environment} chat ({auth_status})")

        except Exception as e:
            logger.exception("Failed to connect to chat")
            if "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
                raise ChatAuthenticationError(f"Chat authentication failed: {e}") from e
            raise ChatConnectionError(f"Failed to connect to chat: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from chat and clean up resources."""
        if self._session and self._is_connected:
            try:
                await self._session.close()
                logger.info("Disconnected from chat")
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            finally:
                self._is_connected = False
                self._session = None

    def is_connected(self) -> bool:
        """Check if client is connected to chat."""
        return self._is_connected and self._session is not None

    async def should_respond_to_message(
        self, message: str, sender: str
    ) -> tuple[bool, str, ParsedCommand | None]:
        """Determine if bot should respond to this message.

        Checks for keywords, mentions, and commands based on configuration.

        Args:
            message: The chat message text
            sender: Username of the sender

        Returns:
            tuple: (should_respond, message_type, parsed_command)
                - should_respond: True if bot should respond
                - message_type: "keyword", "mention", or "command"
                - parsed_command: ParsedCommand if it's a command, None otherwise
        """
        # Skip empty messages
        if not message.strip():
            return False, "keyword", None

        # Skip messages from ignored users (including the bot itself)
        if sender.lower() in self._ignore_users:
            logger.debug(f"Ignoring message from ignored user: {sender}")
            return False, "ignored", None

        # Check for commands first (highest priority)
        if self._respond_to_commands:
            parsed_command = parse_command(message, self._bot_name)
            if parsed_command:
                logger.debug(
                    f"Message '{message}' from {sender} is command: {parsed_command.command}"
                )
                return True, "command", parsed_command

        # Check for mentions
        if self._respond_to_mentions and is_mention(message, self._bot_name):
            logger.debug(f"Message '{message}' from {sender} mentions bot")
            return True, "mention", None

        # Apply keyword filters (original behavior)
        for filter_func in self._filters:
            try:
                if filter_func(message, sender):
                    logger.debug(
                        f"Message '{message}' from {sender} matched keyword filter"
                    )
                    return True, "keyword", None
            except Exception as e:
                logger.warning(f"Filter function error: {e}")
                continue

        return False, "keyword", None

    async def add_message_to_context(
        self, message: str, sender: str, is_bot: bool = False
    ) -> None:
        """Add message to rolling context window with structured metadata.

        Maintains a sliding window of recent chat messages for providing
        context to the LLM. Uses enhanced structured context by default.

        Args:
            message: The chat message text
            sender: Username of the sender
            is_bot: Whether this message is from the bot (KEP-001)
        """
        if self._enhanced_context:
            # Enhanced context mode (KEP-001) - use structured ContextMessage
            context_msg = ContextMessage(
                content=message, sender=sender, timestamp=datetime.now(), is_bot=is_bot
            )
            self._context_messages_structured.append(context_msg)

            logger.debug(
                f"Added to enhanced context: {sender}: {message} (is_bot={is_bot}, "
                f"window size: {len(self._context_messages_structured)})"
            )

            # Log memory usage periodically (KEP-001 Increment C)
            if len(self._context_messages_structured) % 10 == 0:
                self._log_context_memory_usage()
        else:
            # Legacy context mode - maintain backward compatibility
            formatted = f"{sender}: {message}"
            self._context_messages_legacy.append(formatted)

            # Maintain window size for legacy mode
            if len(self._context_messages_legacy) > self._context_window_size:
                self._context_messages_legacy.pop(0)

            logger.debug(
                f"Added to legacy context: {formatted} (window size: {len(self._context_messages_legacy)})"
            )

            # Log memory usage periodically for legacy mode too
            if len(self._context_messages_legacy) % 20 == 0:
                self._log_context_memory_usage()

    def get_context(self, current_sender: str | None = None) -> str | None:
        """Get current conversation context with mention-aware prioritization.

        Implements KEP-001 Increment B by prioritizing messages from users who
        mention the bot and applying token-based context limits.

        Optimized for performance in KEP-001 Increment C with efficient string
        operations and minimal memory allocations.

        Args:
            current_sender: Username of current message sender for prioritization

        Returns:
            str | None: Formatted context or None if no context available
        """
        if self._enhanced_context:
            # Enhanced context mode (KEP-001 Increment B + C optimization)
            if not self._context_messages_structured:
                return None

            # Filter messages efficiently using list comprehension (Increment C optimization)
            if self._include_bot_responses:
                filtered_messages = list(self._context_messages_structured)
            else:
                filtered_messages = [
                    msg for msg in self._context_messages_structured if not msg.is_bot
                ]

            if not filtered_messages:
                return None

            # Apply mention-aware prioritization if enabled
            if self._prioritize_mentions and current_sender:
                prioritized_messages = self._prioritize_context_messages(
                    filtered_messages, current_sender
                )
            else:
                # Use reversed view for efficiency instead of list creation
                prioritized_messages = list(reversed(filtered_messages))

            # Apply token-based context limiting
            context_messages = self._apply_token_limit(prioritized_messages)

            # Optimized string formatting (KEP-001 Increment C)
            # Pre-compute bot prefix to avoid repeated checks and string operations
            formatted_parts = []
            for ctx_msg in context_messages:
                # Use f-string with conditional expression for optimal performance
                formatted_part = (
                    f"🤖 {ctx_msg.sender}: {ctx_msg.content}"
                    if ctx_msg.is_bot
                    else f"{ctx_msg.sender}: {ctx_msg.content}"
                )
                formatted_parts.append(formatted_part)

            # Use join with pre-allocated list for efficiency
            return "\n".join(formatted_parts)
        else:
            # Legacy context mode - maintain exact existing behavior
            if not self._context_messages_legacy:
                return None

            return "\n".join(self._context_messages_legacy)

    def _prioritize_context_messages(
        self, messages: list[ContextMessage], current_sender: str
    ) -> list[ContextMessage]:
        """Prioritize context messages based on mentions and current sender.

        KEP-001 Increment B: Implements mention-aware context prioritization
        by putting messages from the current sender and recent mentions first.

        Optimized for performance in KEP-001 Increment C with efficient message
        categorization and minimal string operations.

        Args:
            messages: List of context messages to prioritize
            current_sender: Username of current message sender

        Returns:
            list[ContextMessage]: Prioritized messages (most relevant first)
        """
        # Pre-compute mention patterns for efficiency (KEP-001 Increment C)
        bot_name_lower = self._bot_name.lower()
        mention_pattern = f"@{bot_name_lower}"

        # Efficient message categorization using list comprehensions
        sender_messages = [msg for msg in messages if msg.sender == current_sender]

        # For non-sender messages, check for mentions efficiently
        non_sender_messages = [msg for msg in messages if msg.sender != current_sender]

        mention_messages = []
        other_messages = []

        for msg in non_sender_messages:
            content_lower = msg.content.lower()
            if bot_name_lower in content_lower or mention_pattern in content_lower:
                mention_messages.append(msg)
            else:
                other_messages.append(msg)

        # Sort each category by timestamp (most recent first) - in-place for efficiency
        sender_messages.sort(key=lambda x: x.timestamp, reverse=True)
        mention_messages.sort(key=lambda x: x.timestamp, reverse=True)
        other_messages.sort(key=lambda x: x.timestamp, reverse=True)

        # Combine prioritized: sender messages, then mentions, then others
        return sender_messages + mention_messages + other_messages

    def _apply_token_limit(
        self, messages: list[ContextMessage]
    ) -> list[ContextMessage]:
        """Apply token-based context limiting to stay within configured limits.

        KEP-001 Increment B: Implements token-aware context truncation to prevent
        overwhelming the LLM with too much context.

        Optimized for performance in KEP-001 Increment C with efficient token
        calculation avoiding unnecessary string operations.

        Args:
            messages: Prioritized messages to potentially truncate

        Returns:
            list[ContextMessage]: Messages within token limit
        """
        if not messages:
            return []

        # Optimized token estimation (KEP-001 Increment C)
        # Avoid string formatting for performance - just calculate lengths
        estimated_tokens = 0
        selected_messages = []

        # Pre-calculate bot prefix length for efficiency
        bot_prefix_length = 5  # Length of "🤖 " (emoji + space)

        for msg in messages:
            # Efficient token estimation without string formatting
            prefix_length = bot_prefix_length if msg.is_bot else 0
            message_length = (
                prefix_length + len(msg.sender) + len(msg.content) + 2  # ": " separator
            )
            message_tokens = message_length // 4  # Conservative token estimation

            # Check if adding this message would exceed limit
            if estimated_tokens + message_tokens > self._max_context_tokens:
                break

            selected_messages.append(msg)
            estimated_tokens += message_tokens

        return selected_messages

    def _get_context_memory_usage(self) -> dict[str, int]:
        """Get detailed memory usage statistics for context storage.

        KEP-001 Increment C: Enhanced monitoring for context memory usage
        to ensure optimal performance and memory efficiency.

        Returns:
            dict[str, int]: Memory usage statistics in bytes
        """
        stats = {
            "total_context_bytes": 0,
            "message_count": 0,
            "avg_message_size": 0,
            "estimated_tokens": 0,
        }

        if self._enhanced_context and hasattr(self, "_context_messages_structured"):
            messages = list(self._context_messages_structured)
            stats["message_count"] = len(messages)

            if messages:
                total_bytes = 0
                total_tokens = 0

                for msg in messages:
                    # Calculate memory usage for ContextMessage
                    content_bytes = sys.getsizeof(msg.content)
                    sender_bytes = sys.getsizeof(msg.sender)
                    timestamp_bytes = sys.getsizeof(msg.timestamp)
                    message_id_bytes = sys.getsizeof(msg.message_id)

                    message_bytes = (
                        content_bytes
                        + sender_bytes
                        + timestamp_bytes
                        + message_id_bytes
                        + sys.getsizeof(msg)  # Object overhead
                    )
                    total_bytes += message_bytes

                    # Estimate tokens (conservative 4 chars per token)
                    total_tokens += len(msg.content) // 4

                stats["total_context_bytes"] = total_bytes
                stats["avg_message_size"] = total_bytes // len(messages)
                stats["estimated_tokens"] = total_tokens

        elif hasattr(self, "_context_messages_legacy"):
            # Legacy mode memory calculation
            legacy_messages = self._context_messages_legacy
            stats["message_count"] = len(legacy_messages)

            if legacy_messages:
                total_bytes = sum(sys.getsizeof(msg) for msg in legacy_messages)
                stats["total_context_bytes"] = total_bytes
                stats["avg_message_size"] = total_bytes // len(legacy_messages)
                stats["estimated_tokens"] = (
                    sum(len(msg) for msg in legacy_messages) // 4
                )

        return stats

    def _log_context_memory_usage(self) -> None:
        """Log context memory usage for monitoring and optimization.

        KEP-001 Increment C: Periodic memory usage logging to track
        performance and identify potential memory issues.
        """
        stats = self._get_context_memory_usage()

        # Log memory usage with appropriate level based on usage
        if stats["total_context_bytes"] > 1024 * 1024:  # > 1MB
            log_level = logging.WARNING
            status = "HIGH"
        elif stats["total_context_bytes"] > 512 * 1024:  # > 512KB
            log_level = logging.INFO
            status = "MODERATE"
        else:
            log_level = logging.DEBUG
            status = "NORMAL"

        logger.log(
            log_level,
            f"Context memory usage [{status}]: "
            f"{stats['total_context_bytes']:,} bytes, "
            f"{stats['message_count']} messages, "
            f"~{stats['estimated_tokens']} tokens, "
            f"avg {stats['avg_message_size']} bytes/msg",
        )

        # Additional warning for excessive memory usage
        if stats["total_context_bytes"] > 5 * 1024 * 1024:  # > 5MB
            logger.warning(
                "Context memory usage is very high (>5MB). Consider reducing "
                "CONTEXT_WINDOW_SIZE or MAX_CONTEXT_TOKENS for better performance."
            )

    def _log_context_performance_metrics(
        self,
        operation: str,
        duration_ms: float,
        message_count: int,
        result_size: int = 0,
    ) -> None:
        """Log performance metrics for context operations.

        KEP-001 Increment C: Track and log performance metrics for context
        operations to identify bottlenecks and optimization opportunities.

        Args:
            operation: Name of the context operation
            duration_ms: Duration in milliseconds
            message_count: Number of messages processed
            result_size: Size of result (bytes, tokens, etc.)
        """
        # Determine log level based on performance
        if duration_ms > 100:  # > 100ms
            log_level = logging.WARNING
            status = "SLOW"
        elif duration_ms > 50:  # > 50ms
            log_level = logging.INFO
            status = "MODERATE"
        else:
            log_level = logging.DEBUG
            status = "FAST"

        logger.log(
            log_level,
            f"Context {operation} performance [{status}]: "
            f"{duration_ms:.2f}ms, {message_count} messages"
            + (f", {result_size} result size" if result_size > 0 else ""),
        )

        # Warning for very slow operations
        if duration_ms > 200:  # > 200ms
            logger.warning(
                f"Context {operation} is very slow ({duration_ms:.2f}ms). "
                "Consider optimizing context window size or token limits."
            )

    def get_context_with_metrics(self, current_sender: str | None = None) -> str | None:
        """Get context with performance metrics tracking.

        KEP-001 Increment C: Enhanced version of get_context that tracks
        performance metrics for monitoring and optimization.

        Args:
            current_sender: Username of current message sender for prioritization

        Returns:
            str | None: Formatted context or None if no context available
        """
        start_time = time.perf_counter()

        # Call the optimized get_context method
        result = self.get_context(current_sender)

        # Calculate performance metrics
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Determine message count based on mode
        if self._enhanced_context and hasattr(self, "_context_messages_structured"):
            message_count = len(self._context_messages_structured)
        elif hasattr(self, "_context_messages_legacy"):
            message_count = len(self._context_messages_legacy)
        else:
            message_count = 0

        # Calculate result size if context exists
        result_size = len(result) if result else 0

        # Log performance metrics
        self._log_context_performance_metrics(
            "get_context", duration_ms, message_count, result_size
        )

        return result

    def _prioritize_context_messages_with_metrics(
        self, messages: list[ContextMessage], current_sender: str
    ) -> list[ContextMessage]:
        """Prioritize context messages with performance tracking.

        KEP-001 Increment C: Enhanced version with performance metrics.

        Args:
            messages: List of context messages to prioritize
            current_sender: Username of current message sender

        Returns:
            list[ContextMessage]: Prioritized messages (most relevant first)
        """
        start_time = time.perf_counter()

        result = self._prioritize_context_messages(messages, current_sender)

        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        self._log_context_performance_metrics(
            "prioritize_messages", duration_ms, len(messages), len(result)
        )

        return result

    def _apply_token_limit_with_metrics(
        self, messages: list[ContextMessage]
    ) -> list[ContextMessage]:
        """Apply token limit with performance tracking.

        KEP-001 Increment C: Enhanced version with performance metrics.

        Args:
            messages: Prioritized messages to potentially truncate

        Returns:
            list[ContextMessage]: Messages within token limit
        """
        start_time = time.perf_counter()

        result = self._apply_token_limit(messages)

        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Calculate estimated tokens for the result
        estimated_tokens = sum(
            (len(msg.sender) + len(msg.content) + 7) // 4  # Conservative estimate
            for msg in result
        )

        self._log_context_performance_metrics(
            "apply_token_limit", duration_ms, len(messages), estimated_tokens
        )

        return result

    def _validate_memory_bounds(self) -> dict[str, bool]:
        """Validate that context memory usage is within acceptable bounds.

        KEP-001 Increment C: Comprehensive memory validation to ensure
        optimal performance and prevent memory issues.

        Returns:
            dict[str, bool]: Validation results for different memory aspects
        """
        stats = self._get_context_memory_usage()

        validation_results = {
            "within_reasonable_bounds": stats["total_context_bytes"]
            < 5 * 1024 * 1024,  # < 5MB
            "within_warning_bounds": stats["total_context_bytes"]
            < 1024 * 1024,  # < 1MB
            "message_count_reasonable": stats["message_count"]
            <= self._context_window_size,
            "avg_message_size_reasonable": stats["avg_message_size"]
            < 10 * 1024,  # < 10KB per message
            "token_count_reasonable": stats["estimated_tokens"]
            <= self._max_context_tokens * 2,  # Allow some buffer
        }

        # Log validation results if there are issues
        failed_validations = [k for k, v in validation_results.items() if not v]
        if failed_validations:
            logger.warning(
                f"Memory validation failed for: {', '.join(failed_validations)}. "
                f"Current stats: {stats}"
            )

        return validation_results

    def _enforce_memory_bounds(self) -> None:
        """Enforce memory bounds by cleaning up context if necessary.

        KEP-001 Increment C: Proactive memory management to prevent
        excessive memory usage and maintain performance.
        """
        validation = self._validate_memory_bounds()

        # If memory usage is excessive, take corrective action
        if not validation["within_reasonable_bounds"]:
            logger.warning(
                "Context memory usage exceeded reasonable bounds. Cleaning up..."
            )

            if self._enhanced_context and hasattr(self, "_context_messages_structured"):
                # Reduce context window size temporarily
                target_size = max(self._context_window_size // 2, 5)

                # Keep only the most recent messages
                recent_messages = list(self._context_messages_structured)[-target_size:]
                self._context_messages_structured.clear()
                self._context_messages_structured.extend(recent_messages)

                logger.info(
                    f"Reduced context to {len(recent_messages)} messages for memory efficiency"
                )

            elif hasattr(self, "_context_messages_legacy"):
                # Reduce legacy context
                target_size = max(self._context_window_size // 2, 5)
                self._context_messages_legacy = self._context_messages_legacy[
                    -target_size:
                ]

                logger.info(
                    f"Reduced legacy context to {len(self._context_messages_legacy)} messages"
                )

    async def add_message_to_context_with_validation(
        self, message: str, sender: str, is_bot: bool = False
    ) -> None:
        """Add message to context with memory validation and bounds enforcement.

        KEP-001 Increment C: Enhanced version that includes memory validation
        and automatic bounds enforcement for production safety.

        Args:
            message: The chat message text
            sender: Username of the sender
            is_bot: Whether this message is from the bot
        """
        # Add message normally
        await self.add_message_to_context(message, sender, is_bot)

        # Validate memory bounds every 25 messages for efficiency
        if self._enhanced_context and hasattr(self, "_context_messages_structured"):
            message_count = len(self._context_messages_structured)
        elif hasattr(self, "_context_messages_legacy"):
            message_count = len(self._context_messages_legacy)
        else:
            message_count = 0

        if message_count % 25 == 0:
            self._enforce_memory_bounds()

    async def send_message(self, message: str) -> None:
        """Send message to chat.

        Args:
            message: Message text to send

        Raises:
            ChatConnectionError: When not connected or send fails
        """
        if not self._session or not self._is_connected:
            raise ChatConnectionError("Not connected to chat")

        try:
            # Truncate very long messages to avoid chat limits
            max_length = 500  # Conservative limit for chat messages
            if len(message) > max_length:
                message = message[: max_length - 3] + "..."
                logger.warning(f"Truncated long message to {max_length} characters")

            await self._session.send_message(message)
            logger.info(f"Sent message: {message[:100]}...")  # Log first 100 chars

        except Exception as e:
            logger.exception("Failed to send message")
            raise ChatConnectionError(f"Failed to send message: {e}") from e

    async def get_next_message(self) -> ProcessedMessage:
        """Get the next message from the processing queue.

        This is a blocking call that waits for messages to be available.

        Returns:
            ProcessedMessage: Next message ready for LLM processing
        """
        return await self._processing_queue.get()

    def _on_message(self, message: Message, _session: AsyncSession) -> None:
        """Handle incoming chat message.

        Called by wsggpy when a new message is received. Processes the message
        and adds it to context, then queues for LLM processing if appropriate.

        Args:
            message: The incoming chat message
            session: The wsggpy session (unused)
        """
        try:
            sender = message.sender.nick
            content = message.message

            # Handle timestamp conversion - could be datetime or numeric
            if isinstance(message.timestamp, datetime):
                # Convert datetime to Unix timestamp in seconds
                timestamp = message.timestamp.timestamp()
            else:
                # Assume it's milliseconds, convert to seconds
                timestamp = message.timestamp / 1000.0

            logger.debug(f"Received message from {sender}: {content}")

            # Add to context (non-blocking)
            _context_task = asyncio.create_task(  # noqa: RUF006
                self.add_message_to_context(content, sender, is_bot=False)
            )

            # Check if we should respond (non-blocking check)
            should_respond_task = asyncio.create_task(
                self.should_respond_to_message(content, sender)
            )

            # Queue for processing if filters match
            _queue_task = asyncio.create_task(  # noqa: RUF006
                self._maybe_queue_message(
                    content, sender, timestamp, should_respond_task
                )
            )

        except Exception:
            logger.exception("Error processing message")

    async def _maybe_queue_message(
        self,
        content: str,
        sender: str,
        timestamp: float,
        should_respond_task: asyncio.Task[tuple[bool, str, ParsedCommand | None]],
    ) -> None:
        """Queue message for LLM processing if filters match.

        Args:
            content: Message content
            sender: Message sender
            timestamp: Message timestamp
            should_respond_task: Task that determines if we should respond
        """
        try:
            should_respond, message_type, parsed_command = await should_respond_task
            if should_respond:
                context = self.get_context(current_sender=sender)

                # Create processed message with new fields
                processed_msg = ProcessedMessage(
                    original_message=content,
                    sender=sender,
                    timestamp=timestamp,
                    context=context,
                    message_type=message_type,
                    command=parsed_command.command if parsed_command else None,
                    command_args=parsed_command.args if parsed_command else None,
                )

                # Queue for LLM processing
                await self._processing_queue.put(processed_msg)

                # Enhanced logging
                if message_type == "command" and parsed_command:
                    logger.info(
                        f"Queued command '{parsed_command.command}' from {sender} for LLM processing"
                    )
                elif message_type == "mention":
                    logger.info(f"Queued mention from {sender} for LLM processing")
                else:
                    logger.info(
                        f"Queued keyword message from {sender} for LLM processing"
                    )

        except Exception:
            logger.exception("Error queueing message")

    def _on_join(self, event: RoomAction, _session: AsyncSession) -> None:
        """Handle user join event.

        Args:
            event: The join event
            session: The wsggpy session (unused)
        """
        username = event.user.nick
        logger.debug(f"👋 {username} joined the chat")

    def _on_quit(self, event: RoomAction, _session: AsyncSession) -> None:
        """Handle user quit event.

        Args:
            event: The quit event
            session: The wsggpy session (unused)
        """
        username = event.user.nick
        logger.debug(f"👋 {username} left the chat")


def create_default_filters(keywords: list[str]) -> list[Callable[[str, str], bool]]:
    """Create default message filters based on keywords.

    Args:
        keywords: List of keywords that should trigger bot responses

    Returns:
        list[Callable]: List of filter functions

    Example:
        >>> filters = create_default_filters(["!bot", "!ask"])
        >>> assert filters[0]("!bot hello", "user123") is True
        >>> assert filters[0]("hello", "user123") is False
    """

    def keyword_filter(message: str, _sender: str) -> bool:
        """Filter based on message starting with specific keywords."""
        message_lower = message.lower().strip()
        return any(message_lower.startswith(keyword.lower()) for keyword in keywords)

    return [keyword_filter]


def create_enhanced_filters(
    keywords: list[str],
    _bot_name: str,
    _respond_to_mentions: bool = True,
    _respond_to_commands: bool = True,
) -> list[Callable[[str, str], bool]]:
    """Create enhanced message filters with mention and command support.

    Note: This function is kept for compatibility, but the new ChatClient
    handles mentions and commands directly in should_respond_to_message().
    This function only creates keyword filters.

    Args:
        keywords: List of keywords that should trigger bot responses
        bot_name: Bot name for mention detection (unused in this implementation)
        respond_to_mentions: Whether to respond to mentions (unused in this implementation)
        respond_to_commands: Whether to respond to commands (unused in this implementation)

    Returns:
        list[Callable]: List of filter functions
    """
    # For backward compatibility, just return keyword filters
    # Mentions and commands are handled in ChatClient.should_respond_to_message()
    return create_default_filters(keywords)
