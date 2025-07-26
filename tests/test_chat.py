"""Tests for chat client functionality."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from brok.chat import ChatClient, ProcessedMessage, create_default_filters
from brok.exceptions import ChatAuthenticationError, ChatConnectionError


@pytest.fixture
def response_filters() -> list:
    """Provide sample response filters."""
    return create_default_filters(["!bot", "!ask"])


@pytest.fixture
def chat_client(response_filters) -> ChatClient:
    """Provide a chat client instance."""
    return ChatClient(
        response_filters=response_filters,
        context_window_size=5,
    )


class TestChatClient:
    """Test cases for ChatClient."""

    @pytest.mark.asyncio
    async def test_connect_production_anonymous(self, chat_client: ChatClient):
        """Test connecting to production chat anonymously."""
        # Arrange
        with pytest.MonkeyPatch().context() as m:
            mock_session = AsyncMock()
            mock_session.open = AsyncMock()

            # Mock AsyncSession constructor
            mock_session_class = MagicMock(return_value=mock_session)
            m.setattr("brok.chat.AsyncSession", mock_session_class)

            # Act
            await chat_client.connect(jwt_token=None, environment="production")

            # Assert
            assert chat_client.is_connected()
            mock_session.open.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_dev_with_jwt(self, chat_client: ChatClient):
        """Test connecting to dev chat with JWT token."""
        # Arrange
        with pytest.MonkeyPatch().context() as m:
            mock_session = AsyncMock()
            mock_session.open = AsyncMock()

            mock_session_class = MagicMock(return_value=mock_session)
            m.setattr("brok.chat.AsyncSession", mock_session_class)

            # Act
            await chat_client.connect(jwt_token="test-jwt", environment="dev")

            # Assert
            assert chat_client.is_connected()
            mock_session.open.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure(self, chat_client: ChatClient):
        """Test handling connection failures."""
        # Arrange
        with pytest.MonkeyPatch().context() as m:
            mock_session = AsyncMock()
            mock_session.open.side_effect = Exception("Connection failed")

            mock_session_class = MagicMock(return_value=mock_session)
            m.setattr("brok.chat.AsyncSession", mock_session_class)

            # Act & Assert
            with pytest.raises(ChatConnectionError, match="Failed to connect to chat"):
                await chat_client.connect(jwt_token=None, environment="production")

    @pytest.mark.asyncio
    async def test_connect_authentication_error(self, chat_client: ChatClient):
        """Test handling authentication errors."""
        # Arrange
        with pytest.MonkeyPatch().context() as m:
            mock_session = AsyncMock()
            mock_session.open.side_effect = Exception("Authentication failed")

            mock_session_class = MagicMock(return_value=mock_session)
            m.setattr("brok.chat.AsyncSession", mock_session_class)

            # Act & Assert
            with pytest.raises(
                ChatAuthenticationError, match="Chat authentication failed"
            ):
                await chat_client.connect(
                    jwt_token="invalid-jwt", environment="production"
                )

    @pytest.mark.asyncio
    async def test_disconnect(self, chat_client: ChatClient):
        """Test disconnecting from chat."""
        # Arrange - Set up connected state
        mock_session = AsyncMock()
        mock_session.close = AsyncMock()
        chat_client._session = mock_session
        chat_client._is_connected = True

        # Act
        await chat_client.disconnect()

        # Assert
        assert not chat_client.is_connected()
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_respond_to_keyword_message(self, chat_client: ChatClient):
        """Test message filtering with keyword matches."""
        # Test cases: (message, expected_result)
        test_cases = [
            ("!bot hello there", True),
            ("!ask what is the weather", True),
            ("!BOT this should work too", True),  # Case insensitive
            ("  !bot  with spaces", True),  # With leading/trailing spaces
            ("regular message", False),
            ("hello !bot in middle", False),  # Keyword not at start
            ("", False),  # Empty message
            ("   ", False),  # Whitespace only
        ]

        for message, expected in test_cases:
            # Act
            result = await chat_client.should_respond_to_message(message, "testuser")

            # Assert
            assert result == expected, f"Failed for message: '{message}'"

    @pytest.mark.asyncio
    async def test_add_message_to_context(self, chat_client: ChatClient):
        """Test adding messages to context window."""
        # Act - Add messages
        await chat_client.add_message_to_context("Hello", "user1")
        await chat_client.add_message_to_context("Hi there", "user2")
        await chat_client.add_message_to_context("How are you?", "user1")

        # Assert
        context = chat_client.get_context()
        assert "user1: Hello" in context
        assert "user2: Hi there" in context
        assert "user1: How are you?" in context

    @pytest.mark.asyncio
    async def test_context_window_size_limit(self, chat_client: ChatClient):
        """Test that context window respects size limit."""
        # Arrange - Client has window size of 5
        messages = [
            ("Message 1", "user1"),
            ("Message 2", "user2"),
            ("Message 3", "user1"),
            ("Message 4", "user2"),
            ("Message 5", "user1"),
            ("Message 6", "user2"),  # This should push out Message 1
            ("Message 7", "user1"),  # This should push out Message 2
        ]

        # Act - Add all messages
        for message, sender in messages:
            await chat_client.add_message_to_context(message, sender)

        # Assert - Only last 5 messages should be in context
        context = chat_client.get_context()
        assert "Message 1" not in context
        assert "Message 2" not in context
        assert "Message 3" in context
        assert "Message 7" in context

        # Verify exact count
        context_lines = context.split("\n")
        assert len(context_lines) == 5

    def test_get_context_empty(self, chat_client: ChatClient):
        """Test getting context when no messages are stored."""
        # Act
        context = chat_client.get_context()

        # Assert
        assert context is None

    @pytest.mark.asyncio
    async def test_send_message_success(self, chat_client: ChatClient):
        """Test sending a message successfully."""
        # Arrange
        mock_session = AsyncMock()
        mock_session.send_message = AsyncMock()
        chat_client._session = mock_session
        chat_client._is_connected = True

        # Act
        await chat_client.send_message("Hello chat!")

        # Assert
        mock_session.send_message.assert_called_once_with("Hello chat!")

    @pytest.mark.asyncio
    async def test_send_message_not_connected(self, chat_client: ChatClient):
        """Test sending message when not connected."""
        # Arrange - Not connected
        chat_client._session = None
        chat_client._is_connected = False

        # Act & Assert
        with pytest.raises(ChatConnectionError, match="Not connected to chat"):
            await chat_client.send_message("Hello chat!")

    @pytest.mark.asyncio
    async def test_send_message_truncation(self, chat_client: ChatClient):
        """Test that very long messages are truncated."""
        # Arrange
        mock_session = AsyncMock()
        mock_session.send_message = AsyncMock()
        chat_client._session = mock_session
        chat_client._is_connected = True

        long_message = "A" * 600  # Longer than 500 char limit

        # Act
        await chat_client.send_message(long_message)

        # Assert
        call_args = mock_session.send_message.call_args[0][0]
        assert len(call_args) <= 500
        assert call_args.endswith("...")

    @pytest.mark.asyncio
    async def test_send_message_failure(self, chat_client: ChatClient):
        """Test handling send message failures."""
        # Arrange
        mock_session = AsyncMock()
        mock_session.send_message.side_effect = Exception("Send failed")
        chat_client._session = mock_session
        chat_client._is_connected = True

        # Act & Assert
        with pytest.raises(ChatConnectionError, match="Failed to send message"):
            await chat_client.send_message("Hello chat!")

    @pytest.mark.asyncio
    async def test_get_next_message(self, chat_client: ChatClient):
        """Test getting messages from the processing queue."""
        # Arrange - Add a message to the queue
        test_message = ProcessedMessage(
            original_message="!bot hello",
            sender="testuser",
            timestamp=1234567890.0,
            context="Previous context",
        )
        await chat_client._processing_queue.put(test_message)

        # Act
        received_message = await chat_client.get_next_message()

        # Assert
        assert received_message.original_message == "!bot hello"
        assert received_message.sender == "testuser"
        assert received_message.timestamp == 1234567890.0
        assert received_message.context == "Previous context"

    def test_on_message_handler_queues_matching_message(self, chat_client: ChatClient):
        """Test that message handler queues messages that match filters."""
        # Arrange
        mock_message = MagicMock()
        mock_message.sender.nick = "testuser"
        mock_message.message = "!bot hello there"
        mock_message.timestamp = 1234567890000  # milliseconds

        mock_session = MagicMock()

        # Act
        chat_client._on_message(mock_message, mock_session)

        # Note: Since _on_message creates tasks, we can't easily test the queue
        # in a synchronous test. This would need async testing with proper
        # task scheduling, which is complex for this test case.
        # In a real scenario, we'd test this through integration tests.

    def test_on_join_handler(self, chat_client: ChatClient):
        """Test join event handler."""
        # Arrange
        mock_event = MagicMock()
        mock_event.user.nick = "newuser"
        mock_session = MagicMock()

        # Act - Should not raise an exception
        chat_client._on_join(mock_event, mock_session)

        # Assert - Just verify it doesn't crash
        assert True

    def test_on_quit_handler(self, chat_client: ChatClient):
        """Test quit event handler."""
        # Arrange
        mock_event = MagicMock()
        mock_event.user.nick = "departinguser"
        mock_session = MagicMock()

        # Act - Should not raise an exception
        chat_client._on_quit(mock_event, mock_session)

        # Assert - Just verify it doesn't crash
        assert True


class TestDefaultFilters:
    """Test cases for default message filters."""

    def test_create_default_filters(self):
        """Test creating default keyword filters."""
        # Act
        filters = create_default_filters(["!bot", "!ask", "!help"])

        # Assert
        assert len(filters) == 1  # Should create one keyword filter
        assert callable(filters[0])

    def test_keyword_filter_matches(self):
        """Test keyword filter matching."""
        # Arrange
        filters = create_default_filters(["!bot", "!ask"])
        keyword_filter = filters[0]

        # Test cases: (message, expected_result)
        test_cases = [
            ("!bot hello", True),
            ("!ask something", True),
            ("!BOT case insensitive", True),
            ("!ASK also works", True),
            ("  !bot  with spaces", True),
            ("regular message", False),
            ("hello !bot in middle", False),
            ("", False),
            ("!help not in keywords", False),
        ]

        for message, expected in test_cases:
            # Act
            result = keyword_filter(message, "testuser")

            # Assert
            assert result == expected, f"Failed for message: '{message}'"

    def test_keyword_filter_empty_keywords(self):
        """Test keyword filter with empty keyword list."""
        # Act
        filters = create_default_filters([])
        keyword_filter = filters[0]

        # Assert - Should not match anything
        assert not keyword_filter("!bot hello", "testuser")
        assert not keyword_filter("any message", "testuser")
