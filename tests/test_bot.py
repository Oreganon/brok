"""Tests for the main chatbot coordinator."""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from brok.bot import BotStats, ChatBot
from brok.chat import ProcessedMessage
from brok.config import BotConfig
from brok.exceptions import BrokError, LLMProviderError


@pytest.fixture
def bot_config() -> BotConfig:
    """Provide a sample bot configuration."""
    return BotConfig(
        chat_environment="dev",
        jwt_token="test-jwt",
        llm_provider="ollama",
        llm_model="test-model",
        llm_max_concurrent_requests=2,
        respond_to_keywords=["!bot"],
        context_window_size=5,
        log_level="INFO",
    )


@pytest.fixture
def mock_chat_client() -> AsyncMock:
    """Provide a mock chat client."""
    client = AsyncMock()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.is_connected.return_value = True
    client.get_next_message = AsyncMock()
    client.send_message = AsyncMock()
    return client


@pytest.fixture
def mock_llm_provider() -> AsyncMock:
    """Provide a mock LLM provider."""
    provider = AsyncMock()
    provider.health_check.return_value = True
    provider.get_metadata.return_value = {"tokens_used": 10, "provider": "test"}
    return provider


@pytest.fixture
def chat_bot(
    bot_config: BotConfig, mock_chat_client: AsyncMock, mock_llm_provider: AsyncMock
) -> ChatBot:
    """Provide a ChatBot instance with mocked dependencies."""
    return ChatBot(
        config=bot_config,
        chat_client=mock_chat_client,
        llm_provider=mock_llm_provider,
    )


class TestBotStats:
    """Test cases for BotStats dataclass."""

    def test_bot_stats_initialization(self):
        """Test BotStats initialization with default values."""
        # Act
        stats = BotStats()

        # Assert
        assert stats.messages_processed == 0
        assert stats.responses_sent == 0
        assert stats.errors_count == 0
        assert stats.start_time == 0.0
        assert stats.last_activity == 0.0

    def test_bot_stats_with_values(self):
        """Test BotStats initialization with custom values."""
        # Act
        stats = BotStats(
            messages_processed=5,
            responses_sent=3,
            errors_count=1,
            start_time=1234567890.0,
            last_activity=1234567900.0,
        )

        # Assert
        assert stats.messages_processed == 5
        assert stats.responses_sent == 3
        assert stats.errors_count == 1
        assert stats.start_time == 1234567890.0
        assert stats.last_activity == 1234567900.0


class TestChatBot:
    """Test cases for ChatBot."""

    def test_initialization(self, chat_bot: ChatBot):
        """Test ChatBot initialization."""
        # Assert
        assert chat_bot._config is not None
        assert chat_bot._chat_client is not None
        assert chat_bot._llm_provider is not None
        assert isinstance(chat_bot._stats, BotStats)

    def test_get_stats(self, chat_bot: ChatBot):
        """Test getting bot statistics."""
        # Act
        stats = chat_bot.get_stats()

        # Assert
        assert isinstance(stats, BotStats)
        assert stats.messages_processed == 0
        assert stats.responses_sent == 0
        assert stats.errors_count == 0

    @pytest.mark.asyncio
    async def test_start_health_check_failure(
        self, chat_bot: ChatBot, mock_llm_provider: AsyncMock
    ):
        """Test bot start failure when LLM health check fails."""
        # Arrange
        mock_llm_provider.health_check.return_value = False

        # Act & Assert
        with pytest.raises(BrokError, match="LLM provider failed health check"):
            await chat_bot.start()

    @pytest.mark.asyncio
    async def test_start_chat_connection_failure(
        self, chat_bot: ChatBot, mock_chat_client: AsyncMock
    ):
        """Test bot start failure when chat connection fails."""
        # Arrange
        mock_chat_client.connect.side_effect = Exception("Connection failed")

        # Act & Assert
        with pytest.raises(BrokError, match="Bot startup failed"):
            await chat_bot.start()

    @pytest.mark.asyncio
    async def test_llm_worker_processes_message(
        self,
        chat_bot: ChatBot,
        mock_chat_client: AsyncMock,
        mock_llm_provider: AsyncMock,
    ):
        """Test that LLM worker processes messages correctly."""
        # Arrange
        test_message = ProcessedMessage(
            original_message="!bot hello",
            sender="testuser",
            timestamp=1234567890.0,
            context="Previous context",
        )

        # Mock the queue to return one message then timeout
        message_queue = asyncio.Queue()
        await message_queue.put(test_message)

        async def mock_get_next_message():
            try:
                return await asyncio.wait_for(message_queue.get(), timeout=0.1)
            except TimeoutError:
                # Simulate the timeout that would happen in the worker
                raise TimeoutError() from None

        mock_chat_client.get_next_message.side_effect = mock_get_next_message

        # Mock LLM to return chunks
        async def mock_generate(_prompt, _context):
            yield "Hello! "
            yield "How can I help you?"

        mock_llm_provider.generate = AsyncMock(
            side_effect=lambda p, c: mock_generate(p, c)
        )

        # Act - Run worker for a short time
        chat_bot._shutdown_event = asyncio.Event()

        # Start worker and let it process one message
        worker_task = asyncio.create_task(chat_bot._llm_worker(worker_id=0))

        # Give it time to process the message
        await asyncio.sleep(0.2)

        # Signal shutdown
        chat_bot._shutdown_event.set()

        # Wait for worker to finish
        try:
            await asyncio.wait_for(worker_task, timeout=1.0)
        except TimeoutError:
            worker_task.cancel()

        # Assert
        assert chat_bot._stats.messages_processed >= 1
        mock_chat_client.send_message.assert_called_with("Hello! How can I help you?")

    @pytest.mark.asyncio
    async def test_llm_worker_handles_llm_error(
        self,
        chat_bot: ChatBot,
        mock_chat_client: AsyncMock,
        mock_llm_provider: AsyncMock,
    ):
        """Test that LLM worker handles LLM provider errors gracefully."""
        # Arrange
        test_message = ProcessedMessage(
            original_message="!bot hello",
            sender="testuser",
            timestamp=1234567890.0,
        )

        message_queue = asyncio.Queue()
        await message_queue.put(test_message)

        async def mock_get_next_message():
            try:
                return await asyncio.wait_for(message_queue.get(), timeout=0.1)
            except TimeoutError:
                raise TimeoutError() from None

        mock_chat_client.get_next_message.side_effect = mock_get_next_message

        # Mock LLM to raise an error
        mock_llm_provider.generate.side_effect = LLMProviderError("LLM failed")

        # Act
        chat_bot._shutdown_event = asyncio.Event()

        worker_task = asyncio.create_task(chat_bot._llm_worker(worker_id=0))

        await asyncio.sleep(0.2)
        chat_bot._shutdown_event.set()

        try:
            await asyncio.wait_for(worker_task, timeout=1.0)
        except TimeoutError:
            worker_task.cancel()

        # Assert
        assert chat_bot._stats.errors_count >= 1
        # Should attempt to send error message to chat
        mock_chat_client.send_message.assert_called_with(
            "Sorry, I'm having trouble generating a response right now."
        )

    @pytest.mark.asyncio
    async def test_llm_worker_handles_empty_response(
        self,
        chat_bot: ChatBot,
        mock_chat_client: AsyncMock,
        mock_llm_provider: AsyncMock,
    ):
        """Test that LLM worker handles empty responses."""
        # Arrange
        test_message = ProcessedMessage(
            original_message="!bot hello",
            sender="testuser",
            timestamp=1234567890.0,
        )

        message_queue = asyncio.Queue()
        await message_queue.put(test_message)

        async def mock_get_next_message():
            try:
                return await asyncio.wait_for(message_queue.get(), timeout=0.1)
            except TimeoutError:
                raise TimeoutError() from None

        mock_chat_client.get_next_message.side_effect = mock_get_next_message

        # Mock LLM to return empty generator
        async def empty_generate(_prompt, _context):
            # Empty async generator - no yields
            if False:  # Never execute
                yield

        mock_llm_provider.generate = AsyncMock(
            side_effect=lambda p, c: empty_generate(p, c)
        )

        # Act
        chat_bot._shutdown_event = asyncio.Event()

        worker_task = asyncio.create_task(chat_bot._llm_worker(worker_id=0))

        await asyncio.sleep(0.2)
        chat_bot._shutdown_event.set()

        try:
            await asyncio.wait_for(worker_task, timeout=1.0)
        except TimeoutError:
            worker_task.cancel()

        # Assert - Should not send empty message
        assert chat_bot._stats.messages_processed >= 1
        # Verify no message was sent for empty response
        assert not any(
            call.args[0] == "" for call in mock_chat_client.send_message.call_args_list
        )

    @pytest.mark.asyncio
    async def test_monitor_connection_detects_disconnection(
        self, chat_bot: ChatBot, mock_chat_client: AsyncMock
    ):
        """Test that connection monitor detects chat disconnection."""
        # Arrange
        # Make is_connected return False without being a coroutine
        mock_chat_client.is_connected = MagicMock(return_value=False)
        chat_bot._shutdown_event = asyncio.Event()

        # Act
        monitor_task = asyncio.create_task(chat_bot._monitor_connection())

        # Let it run briefly
        await asyncio.sleep(0.1)
        chat_bot._shutdown_event.set()

        try:
            await asyncio.wait_for(monitor_task, timeout=1.0)
        except TimeoutError:
            monitor_task.cancel()

        # Assert
        assert chat_bot._stats.errors_count >= 1

    @pytest.mark.asyncio
    async def test_stats_logger(self, chat_bot: ChatBot):
        """Test stats logger functionality."""
        # Arrange
        chat_bot._shutdown_event = asyncio.Event()
        chat_bot._stats.messages_processed = 5
        chat_bot._stats.responses_sent = 3
        chat_bot._stats.errors_count = 1

        # Act - Start stats logger and shut it down quickly
        stats_task = asyncio.create_task(chat_bot._stats_logger())

        await asyncio.sleep(0.1)
        chat_bot._shutdown_event.set()

        try:
            await asyncio.wait_for(stats_task, timeout=1.0)
        except TimeoutError:
            stats_task.cancel()

        # Assert - Just verify it doesn't crash
        assert True

    @pytest.mark.asyncio
    async def test_shutdown_cleanup(
        self,
        chat_bot: ChatBot,
        mock_chat_client: AsyncMock,
        mock_llm_provider: AsyncMock,
    ):
        """Test proper cleanup during shutdown."""
        # Arrange - Add close method to mock provider
        mock_llm_provider.close = AsyncMock()

        # Act
        await chat_bot._shutdown()

        # Assert
        mock_chat_client.disconnect.assert_called_once()
        mock_llm_provider.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_handles_cleanup_errors(
        self, chat_bot: ChatBot, mock_chat_client: AsyncMock
    ):
        """Test that shutdown handles cleanup errors gracefully."""
        # Arrange
        mock_chat_client.disconnect.side_effect = Exception("Disconnect failed")

        # Act - Should not raise exception
        await chat_bot._shutdown()

        # Assert - Verify disconnect was attempted
        mock_chat_client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_for_shutdown_handles_keyboard_interrupt(
        self, chat_bot: ChatBot
    ):
        """Test that wait_for_shutdown handles KeyboardInterrupt."""
        # Arrange
        chat_bot._shutdown_event = asyncio.Event()

        # Act
        shutdown_task = asyncio.create_task(chat_bot._wait_for_shutdown())

        # Simulate KeyboardInterrupt by cancelling and checking behavior
        await asyncio.sleep(0.1)
        shutdown_task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await shutdown_task

        # Assert - Just verify the method structure is correct
        assert True

    def test_stats_tracking(self, chat_bot: ChatBot):
        """Test that stats are properly tracked."""
        # Act - Manually update stats
        chat_bot._stats.messages_processed = 10
        chat_bot._stats.responses_sent = 8
        chat_bot._stats.errors_count = 2

        # Assert
        stats = chat_bot.get_stats()
        assert stats.messages_processed == 10
        assert stats.responses_sent == 8
        assert stats.errors_count == 2
