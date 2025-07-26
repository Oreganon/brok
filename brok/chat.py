"""Chat client wrapper and message processing for strims.gg integration."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import logging
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
        enhanced_context: bool = False,
        max_context_tokens: int = 500,
        prioritize_mentions: bool = True,
        include_bot_responses: bool = True,
    ):
        """Initialize chat client.

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

        # Context storage - always initialize both, choose which to use based on flag
        self._context_messages_structured: deque[ContextMessage] = deque(
            maxlen=context_window_size if enhanced_context else 0
        )
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
        """Add message to rolling context window.

        Maintains a sliding window of recent chat messages for providing
        context to the LLM. Supports both legacy and enhanced context modes.

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

    def get_context(self, current_sender: str | None = None) -> str | None:  # noqa: ARG002
        """Get current conversation context as a formatted string.

        Supports both legacy and enhanced context modes while maintaining
        backward compatibility by returning the same string format.

        Args:
            current_sender: Optional sender for mention prioritization (KEP-001)

        Returns:
            str | None: Formatted context or None if no context available
        """
        if self._enhanced_context:
            # Enhanced context mode (KEP-001) - format structured messages
            if not self._context_messages_structured:
                return None

            # For Increment A, maintain simple behavior - just format messages
            # Future increments will add mention prioritization and token limits
            formatted_messages = []
            for ctx_msg in self._context_messages_structured:
                if not self._include_bot_responses and ctx_msg.is_bot:
                    continue
                formatted_messages.append(f"{ctx_msg.sender}: {ctx_msg.content}")

            return "\n".join(formatted_messages) if formatted_messages else None
        else:
            # Legacy context mode - maintain exact existing behavior
            if not self._context_messages_legacy:
                return None

            return "\n".join(self._context_messages_legacy)

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
                context = self.get_context()

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
                if message_type == "command":
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
