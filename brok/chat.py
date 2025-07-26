"""Chat client wrapper and message processing for strims.gg integration."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable

from wsggpy import AsyncSession, ChatEnvironment, Message, RoomAction

from brok.exceptions import ChatAuthenticationError, ChatConnectionError

logger = logging.getLogger(__name__)


@dataclass
class ProcessedMessage:
    """Represents a processed chat message ready for LLM processing.

    Attributes:
        original_message: The original chat message text
        sender: Username of the message sender
        timestamp: Unix timestamp when message was received
        context: Optional conversation context (recent chat history)
    """

    original_message: str
    sender: str
    timestamp: float
    context: str | None = None


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
    ):
        """Initialize chat client.

        Args:
            response_filters: List of functions to determine if bot should respond
            context_window_size: Number of recent messages to keep for context
        """
        self._filters = response_filters
        self._context_window_size = context_window_size
        self._context_messages: list[str] = []
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
            logger.error(f"Failed to connect to chat: {e}")
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

    async def should_respond_to_message(self, message: str, sender: str) -> bool:
        """Determine if bot should respond to this message.

        Applies all configured filters to determine if the message should
        trigger a bot response.

        Args:
            message: The chat message text
            sender: Username of the sender

        Returns:
            bool: True if bot should respond to this message
        """
        # Skip empty messages
        if not message.strip():
            return False

        # Apply all filters - if any return True, we should respond
        for filter_func in self._filters:
            try:
                if filter_func(message, sender):
                    logger.debug(f"Message '{message}' from {sender} matched filter")
                    return True
            except Exception as e:
                logger.warning(f"Filter function error: {e}")
                continue

        return False

    async def add_message_to_context(self, message: str, sender: str) -> None:
        """Add message to rolling context window.

        Maintains a sliding window of recent chat messages for providing
        context to the LLM.

        Args:
            message: The chat message text
            sender: Username of the sender
        """
        formatted = f"{sender}: {message}"
        self._context_messages.append(formatted)

        # Maintain window size
        if len(self._context_messages) > self._context_window_size:
            self._context_messages.pop(0)

        logger.debug(
            f"Added to context: {formatted} (window size: {len(self._context_messages)})"
        )

    def get_context(self) -> str | None:
        """Get current conversation context as a formatted string.

        Returns:
            str | None: Formatted context or None if no context available
        """
        if not self._context_messages:
            return None

        return "\n".join(self._context_messages)

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
            logger.error(f"Failed to send message: {e}")
            raise ChatConnectionError(f"Failed to send message: {e}") from e

    async def get_next_message(self) -> ProcessedMessage:
        """Get the next message from the processing queue.

        This is a blocking call that waits for messages to be available.

        Returns:
            ProcessedMessage: Next message ready for LLM processing
        """
        return await self._processing_queue.get()

    def _on_message(self, message: Message, session: AsyncSession) -> None:
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
            timestamp = message.timestamp / 1000.0  # Convert to seconds

            logger.debug(f"Received message from {sender}: {content}")

            # Add to context (non-blocking)
            asyncio.create_task(self.add_message_to_context(content, sender))

            # Check if we should respond (non-blocking check)
            should_respond = asyncio.create_task(
                self.should_respond_to_message(content, sender)
            )

            # Queue for processing if filters match
            asyncio.create_task(
                self._maybe_queue_message(content, sender, timestamp, should_respond)
            )

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def _maybe_queue_message(
        self,
        content: str,
        sender: str,
        timestamp: float,
        should_respond_task: asyncio.Task[bool],
    ) -> None:
        """Queue message for LLM processing if filters match.

        Args:
            content: Message content
            sender: Message sender
            timestamp: Message timestamp
            should_respond_task: Task that determines if we should respond
        """
        try:
            should_respond = await should_respond_task
            if should_respond:
                context = self.get_context()
                processed_msg = ProcessedMessage(
                    original_message=content,
                    sender=sender,
                    timestamp=timestamp,
                    context=context,
                )

                # Queue for LLM processing
                await self._processing_queue.put(processed_msg)
                logger.info(f"Queued message from {sender} for LLM processing")

        except Exception as e:
            logger.error(f"Error queueing message: {e}")

    def _on_join(self, event: RoomAction, session: AsyncSession) -> None:
        """Handle user join event.

        Args:
            event: The join event
            session: The wsggpy session (unused)
        """
        username = event.user.nick
        logger.debug(f"👋 {username} joined the chat")

    def _on_quit(self, event: RoomAction, session: AsyncSession) -> None:
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

    def keyword_filter(message: str, sender: str) -> bool:
        """Filter based on message starting with specific keywords."""
        message_lower = message.lower().strip()
        return any(message_lower.startswith(keyword.lower()) for keyword in keywords)

    return [keyword_filter]
