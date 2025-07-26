"""Main chatbot coordinator that orchestrates chat and LLM interactions."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
import logging
import re
import sys
import time
from typing import TYPE_CHECKING, Any

from brok.exceptions import BrokError, LLMProviderError, LLMTimeoutError
from brok.tools import (
    CalculatorTool,
    DateTimeTool,
    ToolParser,
    ToolRegistry,
    WeatherTool,
)

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

        # Performance metrics
        avg_response_time: Average LLM response time in seconds
        total_response_time: Cumulative response time for averaging
        max_response_time: Maximum response time observed
        min_response_time: Minimum response time observed

        # Resource metrics
        current_queue_size: Current message queue depth
        max_queue_size: Maximum queue depth observed

        # Provider-specific metrics
        llm_timeouts: Number of LLM timeout errors
        chat_reconnections: Number of chat reconnection attempts

        # Message type breakdown
        command_messages: Messages parsed as commands
        mention_messages: Messages containing mentions
        keyword_messages: Messages matching keywords only
    """

    # Core metrics
    messages_processed: int = 0
    responses_sent: int = 0
    errors_count: int = 0
    start_time: float = 0.0
    last_activity: float = 0.0

    # Performance metrics
    avg_response_time: float = 0.0
    total_response_time: float = 0.0
    max_response_time: float = 0.0
    min_response_time: float = float("inf")

    # Resource metrics
    current_queue_size: int = 0
    max_queue_size: int = 0

    # Provider-specific metrics
    llm_timeouts: int = 0
    chat_reconnections: int = 0

    # Message type breakdown
    command_messages: int = 0
    mention_messages: int = 0
    keyword_messages: int = 0

    def update_response_time(self, response_time: float) -> None:
        """Update response time statistics.

        Args:
            response_time: Response time in seconds
        """
        self.total_response_time += response_time
        self.max_response_time = max(self.max_response_time, response_time)

        if self.min_response_time == float("inf"):
            self.min_response_time = response_time
        else:
            self.min_response_time = min(self.min_response_time, response_time)

        # Calculate running average
        if self.responses_sent > 0:
            self.avg_response_time = self.total_response_time / self.responses_sent

    def update_queue_size(self, current_size: int) -> None:
        """Update queue size statistics.

        Args:
            current_size: Current queue depth
        """
        self.current_queue_size = current_size
        self.max_queue_size = max(self.max_queue_size, current_size)

    def increment_message_type(self, message_type: str) -> None:
        """Increment counter for specific message type.

        Args:
            message_type: Type of message ("command", "mention", "keyword")
        """
        if message_type == "command":
            self.command_messages += 1
        elif message_type == "mention":
            self.mention_messages += 1
        elif message_type == "keyword":
            self.keyword_messages += 1


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

        # Initialize tool system
        self._tool_registry = ToolRegistry()
        self._tool_parser = ToolParser()
        self._setup_tools()

    def _setup_tools(self) -> None:
        """Initialize and register default tools."""
        if not self._config.enable_tools:
            logger.info("Tools are disabled in configuration")
            return

        try:
            # Register weather tool (no API key needed for wttr.in)
            weather_tool = WeatherTool()
            self._tool_registry.register_tool(weather_tool)

            # Register calculator tool
            calculator_tool = CalculatorTool()
            self._tool_registry.register_tool(calculator_tool)

            # Register datetime tool
            datetime_tool = DateTimeTool()
            self._tool_registry.register_tool(datetime_tool)

            # Set up tool parser with available tools
            available_tools = self._tool_registry.get_available_tools()
            self._tool_parser.update_available_tools(available_tools)

            # Set tool registry in LLM provider
            self._llm_provider.set_tool_registry(self._tool_registry)

            logger.info(
                f"Initialized {len(available_tools)} tools: {', '.join(available_tools)}"
            )

        except Exception:
            logger.exception("Error setting up tools")
            # Continue without tools if setup fails
            logger.warning("Continuing without tool support")

    def _strip_xml_tags(self, response: str) -> str:
        """Strip XML tags from LLM response.

        Removes common XML tags that might leak through from XML prompt formatting.
        This is a best-effort cleanup to handle cases where the LLM includes XML
        markup in its response despite instructions not to.

        Args:
            response: The raw LLM response

        Returns:
            str: Response with XML tags removed
        """
        if not response:
            return response or ""

        # Remove common XML tags that might appear in responses
        # Match opening and closing tags, including self-closing tags
        xml_pattern = r"<[^>]+>"

        # First pass: Remove obvious XML tags
        cleaned = re.sub(xml_pattern, "", response)

        # Second pass: Clean up any remaining XML-like patterns
        # Remove XML declarations, CDATA sections, etc.
        cleaned = re.sub(r"<!\[CDATA\[.*?\]\]>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"<\?xml[^>]*\?>", "", cleaned)
        cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.DOTALL)

        # Third pass: Handle malformed XML that might have spaces
        cleaned = re.sub(r"<\s*[^>]*\s*>", "", cleaned)

        # Clean up extra whitespace that might be left after tag removal
        cleaned = re.sub(r"\n\s*\n", "\n", cleaned)  # Multiple empty lines
        cleaned = cleaned.strip()

        # Log if we actually stripped anything for debugging
        if cleaned != response:
            logger.debug(
                f"Stripped XML tags from response. Original length: {len(response)}, cleaned length: {len(cleaned)}"
            )
            logger.debug(f"First 100 chars of original: {response[:100]}")
            logger.debug(f"First 100 chars of cleaned: {cleaned[:100]}")

        return cleaned

    async def _process_response_with_tools(self, response: str, sender: str) -> str:
        """Process LLM response for tool calls and execute them.

        Args:
            response: The LLM response to process
            sender: Username of the message sender

        Returns:
            str: Final response to send to chat (may include tool results)
        """
        # First, strip any XML tags that might have leaked through
        response = self._strip_xml_tags(response)

        # Log the response for debugging tool parsing issues
        logger.debug(
            f"Processing response for tools from {sender}: {response[:100]}..."
        )

        # Check if the response contains a tool call
        tool_call = self._tool_parser.parse_response(response)

        if not tool_call:
            # Check if response looks like a tool call but wasn't parsed
            if "tool" in response.lower() and "{" in response and "}" in response:
                logger.warning(
                    f"Response from {sender} looks like a tool call but wasn't parsed: {response}"
                )
            # No tool call detected, return original response
            return response

        logger.info(
            f"Tool call detected from {sender}: {tool_call.tool_name} with params {tool_call.parameters}"
        )

        try:
            # Execute the tool
            tool_result = await self._tool_registry.execute_tool(
                tool_call.tool_name, tool_call.parameters
            )

            logger.debug(
                f"Tool {tool_call.tool_name} executed successfully, result length: {len(tool_result)}"
            )

            # Format the final response with tool result
            if "error:" in tool_result.lower():
                # Tool execution failed
                return f"I tried to {tool_call.tool_name} but encountered an issue: {tool_result}"
            else:
                # Tool execution succeeded
                return tool_result

        except KeyError:
            logger.warning(f"Unknown tool requested: {tool_call.tool_name}")
            return f"Sorry, I don't have access to the '{tool_call.tool_name}' tool."

        except Exception as e:
            logger.exception(f"Error executing tool {tool_call.tool_name}")
            return f"Sorry, I encountered an error while trying to help: {e!s}"

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
        Implements exponential backoff for reconnection attempts to avoid
        overwhelming the chat server during outages.

        The monitor will:
        1. Check connection status every 10 seconds
        2. Attempt reconnection with exponential backoff if disconnected
        3. Log all connection events for debugging
        4. Update error statistics for monitoring
        """
        reconnect_delay = self._config.initial_reconnect_delay
        max_reconnect_delay = self._config.max_reconnect_delay
        consecutive_failures = 0
        max_consecutive_failures = self._config.max_reconnect_attempts

        while not self._shutdown_event.is_set():
            try:
                if not self._chat_client.is_connected():
                    logger.warning(
                        f"Chat connection lost! Consecutive failures: {consecutive_failures}"
                    )
                    self._stats.errors_count += 1

                    # Check if we should give up
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(
                            f"Chat reconnection failed {max_consecutive_failures} times. "
                            "Giving up on automatic reconnection."
                        )
                        self._shutdown_event.set()
                        break

                    # Attempt reconnection with exponential backoff
                    logger.info(
                        f"Attempting to reconnect to chat in {reconnect_delay:.1f} seconds..."
                    )
                    await asyncio.sleep(reconnect_delay)

                    if self._shutdown_event.is_set():
                        break

                    try:
                        logger.info("Attempting chat reconnection...")
                        await self._chat_client.connect(
                            jwt_token=self._config.jwt_token,
                            environment=self._config.chat_environment,
                        )

                        logger.info("✅ Chat reconnection successful!")
                        # Reset backoff on successful connection
                        reconnect_delay = self._config.initial_reconnect_delay
                        consecutive_failures = 0
                        self._stats.chat_reconnections += 1

                    except Exception:
                        consecutive_failures += 1
                        logger.exception("Chat reconnection failed")
                        self._stats.errors_count += 1

                        # Increase delay with exponential backoff
                        reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

                # Connection is healthy - reset failure counter
                elif consecutive_failures > 0:
                    logger.debug("Chat connection healthy - resetting failure counter")
                    consecutive_failures = 0
                    reconnect_delay = self._config.initial_reconnect_delay

                # Check again based on configured interval
                await asyncio.sleep(self._config.connection_check_interval)

            except Exception:
                logger.exception("Connection monitor error")
                self._stats.errors_count += 1
                await asyncio.sleep(5)  # Brief pause before retrying

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

                # Update stats with message type tracking
                self._stats.messages_processed += 1
                self._stats.last_activity = time.time()

                # Track message type for analytics
                message_type = getattr(message, "message_type", "keyword")
                self._stats.increment_message_type(message_type)

                # Update queue size statistics
                queue_size = self._chat_client._processing_queue.qsize()
                self._stats.update_queue_size(queue_size)

                # Generate response using LLM with timing
                start_time = time.time()
                try:
                    response_chunks = []
                    async for chunk in self._llm_provider.generate(  # type: ignore[attr-defined]
                        message.original_message,
                        message.context,
                        message.context_messages,
                    ):
                        response_chunks.append(chunk)

                    # Calculate response time
                    response_time = time.time() - start_time

                    # Process response for tool calls and send to chat
                    if response_chunks:
                        full_response = "".join(response_chunks)
                        final_response = await self._process_response_with_tools(
                            full_response, message.sender
                        )

                        if final_response:
                            await self._chat_client.send_message(final_response)
                            self._stats.responses_sent += 1

                        # Update response time statistics
                        self._stats.update_response_time(response_time)

                        # Enhanced logging with performance data
                        metadata = self._llm_provider.get_metadata()
                        if metadata:
                            tokens = metadata.get("tokens_used", 0)
                            logger.info(
                                f"Response sent (worker {worker_id}, {tokens} tokens, "
                                f"{response_time:.2f}s, queue: {queue_size})"
                            )
                        else:
                            logger.info(
                                f"Response sent (worker {worker_id}, {response_time:.2f}s, queue: {queue_size})"
                            )
                    else:
                        logger.warning(
                            f"LLM generated empty response for message from {message.sender} "
                            f"(took {response_time:.2f}s)"
                        )

                except LLMTimeoutError:
                    response_time = time.time() - start_time
                    logger.warning(
                        f"LLM timeout processing message from {message.sender} "
                        f"(after {response_time:.2f}s)"
                    )
                    self._stats.llm_timeouts += 1
                    self._stats.errors_count += 1
                except LLMProviderError:
                    response_time = time.time() - start_time
                    logger.exception(
                        f"LLM error processing message from {message.sender} "
                        f"(after {response_time:.2f}s)"
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
        """Log periodic statistics about bot performance with enhanced metrics."""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(300)  # Log stats every 5 minutes

            if self._shutdown_event.is_set():
                break

            current_time = time.time()
            uptime = current_time - self._stats.start_time

            # Calculate rates
            message_rate = self._stats.messages_processed / uptime if uptime > 0 else 0
            response_rate = self._stats.responses_sent / uptime if uptime > 0 else 0
            error_rate = (
                self._stats.errors_count / self._stats.messages_processed
                if self._stats.messages_processed > 0
                else 0
            )

            # Enhanced logging with performance data
            logger.info(
                f"📊 Bot Performance Stats - "
                f"Uptime: {uptime:.0f}s, "
                f"Messages: {self._stats.messages_processed} ({message_rate:.2f}/s), "
                f"Responses: {self._stats.responses_sent} ({response_rate:.2f}/s), "
                f"Errors: {self._stats.errors_count} ({error_rate:.1%})"
            )

            # Performance metrics
            if self._stats.responses_sent > 0:
                logger.info(
                    f"⚡ Response Times - "
                    f"Avg: {self._stats.avg_response_time:.2f}s, "
                    f"Min: {self._stats.min_response_time:.2f}s, "
                    f"Max: {self._stats.max_response_time:.2f}s"
                )

            # Queue and resource metrics
            logger.info(
                f"📋 Resources - "
                f"Queue: {self._stats.current_queue_size} (max: {self._stats.max_queue_size}), "
                f"Timeouts: {self._stats.llm_timeouts}, "
                f"Reconnections: {self._stats.chat_reconnections}"
            )

            # Message type breakdown
            total_typed = (
                self._stats.command_messages
                + self._stats.mention_messages
                + self._stats.keyword_messages
            )
            if total_typed > 0:
                logger.info(
                    f"💬 Message Types - "
                    f"Commands: {self._stats.command_messages} "
                    f"({self._stats.command_messages / total_typed:.1%}), "
                    f"Mentions: {self._stats.mention_messages} "
                    f"({self._stats.mention_messages / total_typed:.1%}), "
                    f"Keywords: {self._stats.keyword_messages} "
                    f"({self._stats.keyword_messages / total_typed:.1%})"
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

    async def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status for monitoring and debugging.

        Returns detailed health information including:
        - Overall status and uptime
        - Performance statistics
        - Provider health status
        - Connection status
        - Resource utilization metrics

        Returns:
            dict: Comprehensive health status information
        """
        current_time = time.time()
        uptime_seconds = (
            current_time - self._stats.start_time if self._stats.start_time > 0 else 0
        )

        # Calculate additional derived metrics
        message_rate = (
            self._stats.messages_processed / uptime_seconds if uptime_seconds > 0 else 0
        )
        response_rate = (
            self._stats.responses_sent / uptime_seconds if uptime_seconds > 0 else 0
        )
        error_rate = (
            self._stats.errors_count / self._stats.messages_processed
            if self._stats.messages_processed > 0
            else 0
        )

        # Determine overall health status
        is_healthy = (
            self._chat_client.is_connected()
            and error_rate < 0.1  # Less than 10% error rate
            and (current_time - self._stats.last_activity)
            < 600  # Active within 10 minutes
        )

        # Check provider health (async)
        try:
            llm_healthy = await asyncio.wait_for(
                self._llm_provider.health_check(), timeout=5.0
            )
        except Exception:
            llm_healthy = False

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": current_time,
            "uptime_seconds": uptime_seconds,
            "uptime_formatted": f"{uptime_seconds // 3600:.0f}h {(uptime_seconds % 3600) // 60:.0f}m",
            # Connection status
            "connections": {
                "chat_connected": self._chat_client.is_connected(),
                "llm_healthy": llm_healthy,
            },
            # Core statistics
            "statistics": {
                "messages_processed": self._stats.messages_processed,
                "responses_sent": self._stats.responses_sent,
                "errors_count": self._stats.errors_count,
                "last_activity": self._stats.last_activity,
                "time_since_last_activity": current_time - self._stats.last_activity,
            },
            # Performance metrics
            "performance": {
                "avg_response_time": round(self._stats.avg_response_time, 3),
                "min_response_time": round(self._stats.min_response_time, 3)
                if self._stats.min_response_time != float("inf")
                else None,
                "max_response_time": round(self._stats.max_response_time, 3),
                "message_rate_per_second": round(message_rate, 3),
                "response_rate_per_second": round(response_rate, 3),
                "error_rate": round(error_rate, 3),
            },
            # Resource metrics
            "resources": {
                "current_queue_size": self._stats.current_queue_size,
                "max_queue_size": self._stats.max_queue_size,
            },
            # Provider-specific metrics
            "providers": {
                "llm_timeouts": self._stats.llm_timeouts,
                "chat_reconnections": self._stats.chat_reconnections,
            },
            # Message type breakdown
            "message_types": {
                "commands": self._stats.command_messages,
                "mentions": self._stats.mention_messages,
                "keywords": self._stats.keyword_messages,
            },
            # Configuration info
            "config": {
                "llm_provider": self._config.llm_provider,
                "llm_model": self._config.llm_model,
                "chat_environment": self._config.chat_environment,
                "max_concurrent_requests": self._config.llm_max_concurrent_requests,
                "context_window_size": self._config.context_window_size,
            },
            # Version info
            "version": {
                "brok": getattr(self, "__version__", "0.1.0"),
                "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            },
        }
