"""Main chatbot coordinator that orchestrates chat and LLM interactions."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
import logging
import time
from typing import TYPE_CHECKING

from brok.exceptions import BrokError, LLMProviderError

if TYPE_CHECKING:
    from brok.chat import ChatClient
    from brok.config import BotConfig
    from brok.llm.base import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class BotStats:
    """Runtime statistics for monitoring bot performance.

    Attributes:
        messages_processed: Total number of messages processed
        responses_sent: Total number of responses sent to chat
        errors_count: Total number of errors encountered
        start_time: Bot start timestamp
        last_activity: Timestamp of last message processed
    """

    messages_processed: int = 0
    responses_sent: int = 0
    errors_count: int = 0
    start_time: float = 0.0
    last_activity: float = 0.0


class ChatBot:
    """Main chatbot coordinator.

    Orchestrates the interaction between chat client and LLM provider using
    asyncio.TaskGroup for clean task management and graceful shutdown.

    The bot follows this flow:
    1. Connect to chat and start listening for messages
    2. Filter incoming messages based on configured keywords
    3. Queue matching messages for LLM processing
    4. Generate responses using LLM provider
    5. Send responses back to chat

    Example:
        >>> config = BotConfig.from_env()
        >>> chat_client = ChatClient(...)
        >>> llm_provider = OllamaProvider(...)
        >>> bot = ChatBot(config, chat_client, llm_provider)
        >>> await bot.start()  # Runs until interrupted
    """

    def __init__(
        self,
        config: BotConfig,
        chat_client: ChatClient,
        llm_provider: LLMProvider,
    ):
        """Initialize chatbot coordinator.

        Args:
            config: Bot configuration
            chat_client: Chat client for strims.gg integration
            llm_provider: LLM provider for text generation
        """
        self._config = config
        self._chat_client = chat_client
        self._llm_provider = llm_provider
        self._stats = BotStats()
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the chatbot with TaskGroup for clean cancellation.

        Runs the bot until interrupted (Ctrl+C) or an unrecoverable error occurs.
        Uses asyncio.TaskGroup to manage concurrent tasks and ensure clean shutdown.

        Raises:
            BrokError: When bot fails to start or encounters fatal error
        """
        self._stats.start_time = time.time()
        logger.info("Starting brok chatbot...")

        try:
            # First, verify LLM provider is healthy
            if not await self._llm_provider.health_check():
                raise BrokError("LLM provider failed health check - cannot start bot")

            # Connect to chat
            await self._chat_client.connect(
                jwt_token=self._config.jwt_token,
                environment=self._config.chat_environment,
            )

            logger.info(
                f"Bot started successfully! Listening for messages with keywords: {self._config.respond_to_keywords}"
            )
            logger.info("Press Ctrl+C to stop the bot")

            # Start main processing loop with TaskGroup
            try:
                async with asyncio.TaskGroup() as tg:
                    # Start chat connection monitor
                    tg.create_task(self._monitor_connection())

                    # Start LLM processing workers (limited concurrency)
                    for i in range(self._config.llm_max_concurrent_requests):
                        tg.create_task(self._llm_worker(worker_id=i))

                    # Start stats logger
                    tg.create_task(self._stats_logger())

                    # Wait for shutdown signal
                    tg.create_task(self._wait_for_shutdown())

            except* Exception as eg:
                # Handle exception group from TaskGroup
                for e in eg.exceptions:
                    if isinstance(e, KeyboardInterrupt):
                        logger.info("Received interrupt signal")
                    else:
                        logger.exception(f"Task error: {e}")
                        self._stats.errors_count += 1

        except Exception as e:
            logger.exception("Failed to start bot")
            self._stats.errors_count += 1
            raise BrokError(f"Bot startup failed: {e}") from e

        finally:
            await self._shutdown()

    async def _monitor_connection(self) -> None:
        """Monitor chat connection and handle reconnection if needed.

        Runs continuously to ensure chat connection remains active.
        Currently just monitors - reconnection logic can be added later.
        """
        while not self._shutdown_event.is_set():
            try:
                if not self._chat_client.is_connected():
                    logger.warning("Chat connection lost!")
                    self._stats.errors_count += 1
                    # For now, just log - reconnection logic can be added later

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception:
                logger.exception("Connection monitor error")
                await asyncio.sleep(5)

    async def _llm_worker(self, worker_id: int) -> None:
        """Worker to process LLM requests from queue.

        Each worker processes messages sequentially to avoid overwhelming
        the LLM provider. Multiple workers allow some parallelism while
        maintaining resource limits.

        Args:
            worker_id: Unique identifier for this worker
        """
        logger.debug(f"LLM worker {worker_id} started")

        while not self._shutdown_event.is_set():
            try:
                # Get message from queue with timeout
                try:
                    message = await asyncio.wait_for(
                        self._chat_client.get_next_message(), timeout=1.0
                    )
                except TimeoutError:
                    continue  # Check for shutdown and try again

                logger.info(
                    f"Worker {worker_id} processing message from {message.sender}"
                )

                # Update stats
                self._stats.messages_processed += 1
                self._stats.last_activity = time.time()

                # Generate response using LLM
                try:
                    response_chunks = []
                    async for chunk in self._llm_provider.generate(
                        message.original_message, message.context
                    ):
                        response_chunks.append(chunk)

                    # Send complete response to chat
                    if response_chunks:
                        full_response = "".join(response_chunks)
                        await self._chat_client.send_message(full_response)
                        self._stats.responses_sent += 1

                        # Log metadata if available
                        metadata = self._llm_provider.get_metadata()
                        if metadata:
                            tokens = metadata.get("tokens_used", 0)
                            logger.info(
                                f"Response sent (worker {worker_id}, {tokens} tokens)"
                            )
                    else:
                        logger.warning(
                            f"LLM generated empty response for message from {message.sender}"
                        )

                except LLMProviderError:
                    logger.exception(
                        f"LLM error processing message from {message.sender}"
                    )
                    self._stats.errors_count += 1

                    # Send error message to chat (optional)
                    with contextlib.suppress(Exception):
                        await self._chat_client.send_message(
                            "Sorry, I'm having trouble generating a response right now."
                        )

                except Exception:
                    logger.exception(f"Unexpected error in worker {worker_id}")
                    self._stats.errors_count += 1

            except Exception:
                logger.exception(f"Worker {worker_id} error")
                self._stats.errors_count += 1
                await asyncio.sleep(1)  # Brief pause before retrying

        logger.debug(f"LLM worker {worker_id} stopped")

    async def _stats_logger(self) -> None:
        """Log periodic statistics about bot performance."""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(300)  # Log stats every 5 minutes

            if self._shutdown_event.is_set():
                break

            uptime = time.time() - self._stats.start_time
            logger.info(
                f"Bot stats - Uptime: {uptime:.0f}s, "
                f"Messages: {self._stats.messages_processed}, "
                f"Responses: {self._stats.responses_sent}, "
                f"Errors: {self._stats.errors_count}"
            )

    async def _wait_for_shutdown(self) -> None:
        """Wait for shutdown signal (Ctrl+C)."""
        try:
            # Wait indefinitely until interrupted
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            self._shutdown_event.set()

    async def _shutdown(self) -> None:
        """Clean shutdown of all bot components."""
        logger.info("Shutting down bot...")

        # Set shutdown event for all workers
        self._shutdown_event.set()

        try:
            # Disconnect from chat
            await self._chat_client.disconnect()
        except Exception as e:
            logger.warning(f"Error during chat disconnect: {e}")

        try:
            # Close LLM provider if it has a close method
            if hasattr(self._llm_provider, "close"):
                await self._llm_provider.close()
        except Exception as e:
            logger.warning(f"Error during LLM provider cleanup: {e}")

        # Log final stats
        uptime = time.time() - self._stats.start_time
        logger.info(
            f"Bot shutdown complete - Final stats: "
            f"Uptime: {uptime:.0f}s, "
            f"Messages: {self._stats.messages_processed}, "
            f"Responses: {self._stats.responses_sent}, "
            f"Errors: {self._stats.errors_count}"
        )

    def get_stats(self) -> BotStats:
        """Get current bot statistics.

        Returns:
            BotStats: Current runtime statistics
        """
        return self._stats
