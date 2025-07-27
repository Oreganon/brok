"""Tests for chat client functionality."""

from __future__ import annotations

import asyncio
from datetime import datetime
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from wsggpy import AsyncSession, Message, User

from brok.chat import (
    ChatClient,
    ChatStats,
    ContextMessage,
    ProcessedMessage,
    create_default_filters,
    is_mention,
    parse_command,
)
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
        bot_name="testbot",
        respond_to_mentions=True,
        respond_to_commands=True,
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
            # Add sync methods that return proper values
            mock_session.get_connection_info = MagicMock(
                return_value={"connected": True, "reconnecting": False}
            )

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
            # Add sync methods that return proper values
            mock_session.get_connection_info = MagicMock(
                return_value={"connected": True, "reconnecting": False}
            )

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
        # Test cases: (message, expected_should_respond, expected_message_type)
        # Note: !bot and !ask will be parsed as commands, not keywords
        test_cases = [
            ("!bot hello there", True, "command"),  # Commands take priority
            ("!ask what is the weather", True, "command"),  # Commands take priority
            ("!BOT this should work too", True, "command"),  # Case insensitive commands
            ("  !bot  with spaces", True, "command"),  # With leading/trailing spaces
            ("regular message", False, "keyword"),
            ("hello !bot in middle", False, "keyword"),  # Keyword not at start
            ("", False, "keyword"),  # Empty message
            ("   ", False, "keyword"),  # Whitespace only
        ]

        for message, expected_respond, expected_type in test_cases:
            # Act
            (
                should_respond,
                message_type,
                parsed_command,
            ) = await chat_client.should_respond_to_message(message, "testuser")

            # Assert
            assert should_respond == expected_respond, (
                f"Failed should_respond for message: '{message}'"
            )
            if expected_respond:
                assert message_type == expected_type, (
                    f"Failed message_type for message: '{message}'"
                )

    @pytest.mark.asyncio
    async def test_should_respond_to_pure_keywords(self):
        """Test message filtering with pure keyword matches that aren't parsed as commands."""
        # Create a client with keywords that won't be parsed as commands
        filters = create_default_filters(["hey", "hello"])
        pure_keyword_client = ChatClient(
            response_filters=filters,
            context_window_size=5,
            bot_name="testbot",
            respond_to_mentions=True,
            respond_to_commands=True,
        )

        # Test cases: (message, expected_should_respond, expected_message_type)
        test_cases = [
            ("hey there", True, "keyword"),
            ("hello world", True, "keyword"),
            ("HEY case insensitive", True, "keyword"),
            ("  hello  with spaces", True, "keyword"),
            ("regular message", False, "keyword"),
            ("say hey in middle", False, "keyword"),  # Keyword not at start
            ("", False, "keyword"),  # Empty message
        ]

        for message, expected_respond, expected_type in test_cases:
            # Act
            (
                should_respond,
                message_type,
                parsed_command,
            ) = await pure_keyword_client.should_respond_to_message(message, "testuser")

            # Assert
            assert should_respond == expected_respond, (
                f"Failed should_respond for message: '{message}'"
            )
            if expected_respond:
                assert message_type == expected_type, (
                    f"Failed message_type for message: '{message}'"
                )

    @pytest.mark.asyncio
    async def test_should_respond_to_mentions(self, chat_client: ChatClient):
        """Test message filtering with mention detection."""
        # Test cases: (message, expected_should_respond, expected_message_type)
        # Note: Messages with commands take priority over mentions
        test_cases = [
            ("@testbot hello there", True, "command"),  # Commands take priority
            ("testbot: how are you?", True, "command"),  # Commands take priority
            ("testbot, what's up?", True, "command"),  # Commands take priority
            ("testbot tell me something", True, "command"),  # Commands take priority
            ("@testbot", True, "mention"),  # Just mention, no command
            ("@TESTBOT", True, "mention"),  # Case insensitive mention only
            ("testbot:", True, "mention"),  # Just mention with colon, no command
            ("TESTBOT:", True, "mention"),  # Case insensitive mention only
            (
                "hello @testbot in middle",
                True,
                "mention",
            ),  # Mention anywhere, no command
            ("anothertestbot hello", False, "keyword"),  # Wrong bot name
            ("regular message", False, "keyword"),
            ("", False, "keyword"),
        ]

        for message, expected_respond, expected_type in test_cases:
            # Act
            (
                should_respond,
                message_type,
                parsed_command,
            ) = await chat_client.should_respond_to_message(message, "testuser")

            # Assert
            assert should_respond == expected_respond, (
                f"Failed should_respond for message: '{message}'"
            )
            if expected_respond:
                assert message_type == expected_type, (
                    f"Failed message_type for message: '{message}'"
                )

    @pytest.mark.asyncio
    async def test_should_respond_to_commands(self, chat_client: ChatClient):
        """Test message filtering with command parsing."""
        # Test cases: (message, expected_should_respond, expected_message_type, expected_command)
        test_cases = [
            ("!help", True, "command", "help"),
            ("!status check", True, "command", "status"),
            ("@testbot help me", True, "command", "help"),
            ("testbot: status", True, "command", "status"),
            ("testbot help with something", True, "command", "help"),
            ("!HELP case insensitive", True, "command", "help"),
            ("@TESTBOT STATUS", True, "command", "status"),
            ("regular message", False, "keyword", None),
            ("", False, "keyword", None),
        ]

        for message, expected_respond, expected_type, expected_command in test_cases:
            # Act
            (
                should_respond,
                message_type,
                parsed_command,
            ) = await chat_client.should_respond_to_message(message, "testuser")

            # Assert
            assert should_respond == expected_respond, (
                f"Failed should_respond for message: '{message}'"
            )
            if expected_respond and expected_type == "command":
                assert message_type == expected_type, (
                    f"Failed message_type for message: '{message}'"
                )
                assert parsed_command is not None, (
                    f"Expected parsed_command for message: '{message}'"
                )
                assert parsed_command.command == expected_command, (
                    f"Failed command parsing for message: '{message}'"
                )

    @pytest.mark.asyncio
    async def test_command_priority_over_mention(self, chat_client: ChatClient):
        """Test that commands take priority over mentions."""
        # Act
        (
            should_respond,
            message_type,
            parsed_command,
        ) = await chat_client.should_respond_to_message("@testbot help", "testuser")

        # Assert
        assert should_respond is True
        assert message_type == "command"  # Should be command, not mention
        assert parsed_command is not None
        assert parsed_command.command == "help"

    @pytest.mark.asyncio
    async def test_add_message_to_context(self, chat_client: ChatClient):
        """Test adding messages to context window."""
        # Act - Add messages
        await chat_client.add_message_to_context("Hello", "user1", is_bot=False)
        await chat_client.add_message_to_context("Hi there", "user2", is_bot=False)
        await chat_client.add_message_to_context("How are you?", "user1", is_bot=False)

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
            await chat_client.add_message_to_context(message, sender, is_bot=False)

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

    @pytest.mark.asyncio
    async def test_should_ignore_bot_self_messages(self, response_filters):
        """Test that bot ignores its own messages to prevent self-prompting."""
        # Arrange
        chat_client = ChatClient(
            response_filters=response_filters,
            context_window_size=5,
            bot_name="testbot",
            respond_to_mentions=True,
            respond_to_commands=True,
        )

        # Act - bot sends message to itself
        (
            should_respond,
            message_type,
            command,
        ) = await chat_client.should_respond_to_message("Hello everyone!", "testbot")

        # Assert
        assert should_respond is False
        assert message_type == "ignored"
        assert command is None

    @pytest.mark.asyncio
    async def test_should_ignore_configured_users(self, response_filters):
        """Test that bot ignores messages from users in ignore_users list."""
        # Arrange
        chat_client = ChatClient(
            response_filters=response_filters,
            context_window_size=5,
            bot_name="testbot",
            respond_to_mentions=True,
            respond_to_commands=True,
            ignore_users=["spammer", "trolluser"],
        )

        # Act - ignored user sends message
        (
            should_respond,
            message_type,
            command,
        ) = await chat_client.should_respond_to_message("@testbot hello", "spammer")

        # Assert
        assert should_respond is False
        assert message_type == "ignored"
        assert command is None

    @pytest.mark.asyncio
    async def test_ignore_users_case_insensitive(self, response_filters):
        """Test that user ignoring is case-insensitive."""
        # Arrange
        chat_client = ChatClient(
            response_filters=response_filters,
            context_window_size=5,
            bot_name="TestBot",  # Mixed case bot name
            respond_to_mentions=True,
            respond_to_commands=True,
            ignore_users=["SpamUser"],  # Mixed case ignored user
        )

        test_cases = [
            ("testbot", True),  # Bot name in different case
            ("TESTBOT", True),  # Bot name in caps
            ("spamuser", True),  # Ignored user in different case
            ("SPAMUSER", True),  # Ignored user in caps
            ("normaluser", False),  # Normal user should not be ignored
        ]

        for sender, should_be_ignored in test_cases:
            # Act - using a mention format that won't be parsed as command
            (
                should_respond,
                message_type,
                command,
            ) = await chat_client.should_respond_to_message(
                "Hi @TestBot how are you?", sender
            )

            # Assert
            if should_be_ignored:
                assert should_respond is False, f"Expected to ignore {sender}"
                assert message_type == "ignored", (
                    f"Expected 'ignored' type for {sender}"
                )
            else:
                assert should_respond is True, f"Expected to respond to {sender}"
                assert message_type == "mention", (
                    f"Expected 'mention' type for {sender}"
                )

    @pytest.mark.asyncio
    async def test_enhanced_context_feature_flag(self) -> None:
        """Test that enhanced context feature flag works correctly (KEP-001 Increment A)."""

        # Test enhanced context enabled
        enhanced_client = ChatClient(
            response_filters=create_default_filters(["!bot", "!test"]),
            context_window_size=3,
            enhanced_context=True,
            include_bot_responses=True,
        )

        # Add messages
        await enhanced_client.add_message_to_context("Hello", "user1", is_bot=False)
        await enhanced_client.add_message_to_context("Hi back", "brok", is_bot=True)
        await enhanced_client.add_message_to_context(
            "How are you?", "user1", is_bot=False
        )

        # Check that structured storage is used
        assert len(enhanced_client._context_messages_structured) == 3
        assert not hasattr(enhanced_client, "_context_messages_legacy")

        # Check that context output is correctly formatted
        context = enhanced_client.get_context()
        assert context is not None
        assert "user1: Hello" in context
        assert "brok: Hi back" in context
        assert "user1: How are you?" in context

        # Test legacy context (enhanced_context=False)
        legacy_client = ChatClient(
            response_filters=create_default_filters(["!bot", "!test"]),
            context_window_size=3,
            enhanced_context=False,
        )

        # Add same messages
        await legacy_client.add_message_to_context("Hello", "user1", is_bot=False)
        await legacy_client.add_message_to_context("Hi back", "brok", is_bot=True)
        await legacy_client.add_message_to_context(
            "How are you?", "user1", is_bot=False
        )

        # Check that legacy storage is used
        assert not hasattr(legacy_client, "_context_messages_structured")
        assert len(legacy_client._context_messages_legacy) == 3

        # Check that context output is identical (backward compatibility)
        legacy_context = legacy_client.get_context()
        assert legacy_context is not None
        assert "user1: Hello" in legacy_context
        assert "brok: Hi back" in legacy_context
        assert "user1: How are you?" in legacy_context

    @pytest.mark.asyncio
    async def test_enhanced_context_include_bot_responses_setting(self) -> None:
        """Test include_bot_responses setting in enhanced context mode (KEP-001)."""

        # Test with include_bot_responses=False
        client = ChatClient(
            response_filters=create_default_filters(["!bot", "!test"]),
            context_window_size=5,
            enhanced_context=True,
            include_bot_responses=False,
        )

        # Add user and bot messages
        await client.add_message_to_context("Hello", "user1", is_bot=False)
        await client.add_message_to_context("Hi there!", "brok", is_bot=True)
        await client.add_message_to_context("How are you?", "user1", is_bot=False)
        await client.add_message_to_context("I'm doing well!", "brok", is_bot=True)

        # All messages should be stored
        assert len(client._context_messages_structured) == 4

        # But bot messages should be filtered out of context output
        context = client.get_context()
        assert context is not None
        assert "user1: Hello" in context
        assert "user1: How are you?" in context
        assert "brok:" not in context

    @pytest.mark.asyncio
    async def test_mention_aware_context_prioritization(self) -> None:
        """Test KEP-001 Increment B: mention-aware context prioritization."""
        # Create client with enhanced context and mention prioritization enabled
        client = ChatClient(
            response_filters=[],
            context_window_size=10,
            enhanced_context=True,
            prioritize_mentions=True,
        )

        # Add messages: user1 messages, bot mentions, user2 messages
        await client.add_message_to_context("hello everyone", "user1")
        await client.add_message_to_context("@brok what's the weather?", "user2")
        await client.add_message_to_context("It's sunny!", "brok", is_bot=True)
        await client.add_message_to_context("thanks brok!", "user2")
        await client.add_message_to_context("how are you doing?", "user1")

        # Get context with user2 as current sender (who mentioned bot)
        context = client.get_context(current_sender="user2")

        assert context is not None
        lines = context.split("\n")

        # Should prioritize user2's messages first (current sender)
        assert "user2:" in lines[0]  # user2's latest message first
        assert "user2:" in lines[1]  # user2's mention second

        # Verify bot messages are marked with emoji
        bot_lines = [line for line in lines if line.startswith("🤖")]
        assert len(bot_lines) == 1
        assert "🤖 brok: It's sunny!" in bot_lines[0]

    @pytest.mark.asyncio
    async def test_token_based_context_limiting(self) -> None:
        """Test KEP-001 Increment B: token-based context limiting."""
        # Create client with small token limit for testing
        client = ChatClient(
            response_filters=[],
            context_window_size=10,
            enhanced_context=True,
            max_context_tokens=20,  # Very small limit for testing
        )

        # Add messages that would exceed token limit
        await client.add_message_to_context(
            "This is a very long message that should consume many tokens", "user1"
        )
        await client.add_message_to_context(
            "Another long message that would push us over the limit", "user2"
        )
        await client.add_message_to_context("And yet another message", "user3")
        await client.add_message_to_context("Final message", "user4")

        context = client.get_context()

        assert context is not None
        # Should be truncated due to token limit
        lines = context.split("\n")
        assert len(lines) < 4  # Not all messages should be included

    @pytest.mark.asyncio
    async def test_enhanced_context_without_mention_prioritization(self) -> None:
        """Test enhanced context with mention prioritization disabled."""
        client = ChatClient(
            response_filters=[],
            context_window_size=5,
            enhanced_context=True,
            prioritize_mentions=False,  # Disabled
        )

        await client.add_message_to_context("first message", "user1")
        await client.add_message_to_context("@brok second message", "user2")
        await client.add_message_to_context("third message", "user1")

        # Should use chronological order (most recent first) when prioritization disabled
        context = client.get_context(current_sender="user2")

        assert context is not None
        lines = context.split("\n")

        # Should be in reverse chronological order
        assert "user1: third message" in lines[0]
        assert "user2: @brok second message" in lines[1]
        assert "user1: first message" in lines[2]

    @pytest.mark.asyncio
    async def test_enhanced_context_bot_response_filtering(self) -> None:
        """Test enhanced context with bot response filtering."""
        # Test with bot responses excluded
        client = ChatClient(
            response_filters=[],
            context_window_size=5,
            enhanced_context=True,
            include_bot_responses=False,
        )

        await client.add_message_to_context("user message", "user1")
        await client.add_message_to_context("bot response", "brok", is_bot=True)
        await client.add_message_to_context("another user message", "user2")

        context = client.get_context()
        assert context is not None

        # Should not contain bot responses
        assert "bot response" not in context
        assert "user message" in context
        assert "another user message" in context

        # Test with bot responses included
        client._include_bot_responses = True
        context = client.get_context()
        assert context is not None

        # Should contain bot responses with emoji prefix
        assert "🤖 brok: bot response" in context

    @pytest.mark.asyncio
    async def test_enhanced_context_empty_after_filtering(self) -> None:
        """Test enhanced context returns None when empty after filtering."""
        client = ChatClient(
            response_filters=[],
            context_window_size=5,
            enhanced_context=True,
            include_bot_responses=False,  # Exclude bot responses
        )

        # Add only bot messages
        await client.add_message_to_context("bot message 1", "brok", is_bot=True)
        await client.add_message_to_context("bot message 2", "brok", is_bot=True)

        context = client.get_context()
        # Should return None since all messages are filtered out
        assert context is None

    @pytest.mark.asyncio
    async def test_mention_prioritization_helper_method(self) -> None:
        """Test _prioritize_context_messages helper method directly."""
        client = ChatClient(
            response_filters=[],
            enhanced_context=True,
            bot_name="testbot",
        )

        # Create test messages
        messages = [
            ContextMessage("hello", "user1", datetime.now(), False),
            ContextMessage("@testbot help me", "user2", datetime.now(), False),
            ContextMessage("random message", "user3", datetime.now(), False),
            ContextMessage("another message", "user2", datetime.now(), False),
            ContextMessage("testbot, what time is it?", "user1", datetime.now(), False),
        ]

        # Prioritize with user2 as current sender
        prioritized = client._prioritize_context_messages(messages, "user2")

        # user2 messages should come first
        assert prioritized[0].sender == "user2"
        assert prioritized[1].sender == "user2"

        # Then mentions of testbot
        mention_indices = [
            i
            for i, msg in enumerate(prioritized)
            if "testbot" in msg.content.lower() or "@testbot" in msg.content.lower()
        ]
        assert len(mention_indices) > 0

    @pytest.mark.asyncio
    async def test_token_limit_helper_method(self) -> None:
        """Test _apply_token_limit helper method directly."""
        client = ChatClient(
            response_filters=[],
            enhanced_context=True,
            max_context_tokens=10,  # Very small limit for testing
        )

        # Create messages with known content lengths
        messages = [
            ContextMessage("short", "user1", datetime.now(), False),  # ~10 chars
            ContextMessage(
                "medium length message", "user2", datetime.now(), False
            ),  # ~25 chars
            ContextMessage(
                "this is a very long message that exceeds limits",
                "user3",
                datetime.now(),
                False,
            ),  # ~50 chars
        ]

        limited = client._apply_token_limit(messages)

        # Should include some but not all messages due to token limit
        assert len(limited) < len(messages)
        assert len(limited) > 0

    @pytest.mark.asyncio
    async def test_context_performance_with_large_windows(self) -> None:
        """Test KEP-001 Increment C: performance with large context windows."""

        # Create client with large context window for performance testing
        client = ChatClient(
            response_filters=[],
            context_window_size=500,  # Large window
            enhanced_context=True,
            max_context_tokens=2000,  # High token limit
        )

        # Add many messages to test performance
        start_time = time.perf_counter()

        for i in range(500):
            await client.add_message_to_context(
                f"Test message {i} with some content to simulate real chat",
                f"user{i % 10}",  # 10 different users
                is_bot=(i % 5 == 0),  # Every 5th message is from bot
            )

        add_duration = time.perf_counter() - start_time

        # Test context retrieval performance
        start_time = time.perf_counter()
        context = client.get_context(current_sender="user1")
        get_duration = time.perf_counter() - start_time

        # Verify results
        assert context is not None
        assert len(context) > 0

        # Performance assertions (should be fast even with 500 messages)
        assert add_duration < 1.0, (
            f"Adding 500 messages took {add_duration:.3f}s (too slow)"
        )
        assert get_duration < 0.1, (
            f"Getting context took {get_duration:.3f}s (too slow)"
        )

        # Memory usage should be reasonable
        memory_stats = client._get_context_memory_usage()
        assert memory_stats["total_context_bytes"] < 10 * 1024 * 1024  # < 10MB

    @pytest.mark.asyncio
    async def test_prioritization_performance_stress_test(self) -> None:
        """Test KEP-001 Increment C: prioritization performance under stress."""

        client = ChatClient(
            response_filters=[],
            context_window_size=1000,  # Very large window
            enhanced_context=True,
            prioritize_mentions=True,
            bot_name="testbot",
        )

        # Create a mix of regular messages and mentions
        messages = []
        for i in range(1000):
            if i % 20 == 0:  # 5% mention rate
                content = f"@testbot help with task {i}"
            else:
                content = f"Regular message {i} with varied content length for testing"

            msg = ContextMessage(
                content=content,
                sender=f"user{i % 50}",  # 50 different users
                timestamp=datetime.now(),
                is_bot=False,
            )
            messages.append(msg)

        # Test prioritization performance
        start_time = time.perf_counter()
        prioritized = client._prioritize_context_messages(messages, "user1")
        duration = time.perf_counter() - start_time

        # Verify results
        assert len(prioritized) == len(messages)
        assert prioritized[0].sender == "user1"  # user1 messages should be first

        # Performance assertion
        assert duration < 0.05, (
            f"Prioritizing 1000 messages took {duration:.3f}s (too slow)"
        )

    @pytest.mark.asyncio
    async def test_token_limiting_performance_stress_test(self) -> None:
        """Test KEP-001 Increment C: token limiting performance under stress."""

        client = ChatClient(
            response_filters=[],
            enhanced_context=True,
            max_context_tokens=1000,  # Moderate token limit
        )

        # Create messages with varying content lengths
        messages = []
        for i in range(1000):
            # Vary content length to test token calculation efficiency
            content_length = 50 + (i % 200)  # 50-250 character messages
            content = "x" * content_length

            msg = ContextMessage(
                content=content,
                sender=f"user{i % 20}",
                timestamp=datetime.now(),
                is_bot=(i % 10 == 0),
            )
            messages.append(msg)

        # Test token limiting performance
        start_time = time.perf_counter()
        limited = client._apply_token_limit(messages)
        duration = time.perf_counter() - start_time

        # Verify results
        assert len(limited) < len(messages)  # Should be limited
        assert len(limited) > 0  # Should include some messages

        # Performance assertion
        assert duration < 0.02, (
            f"Token limiting 1000 messages took {duration:.3f}s (too slow)"
        )

    @pytest.mark.asyncio
    async def test_memory_usage_validation_under_load(self) -> None:
        """Test KEP-001 Increment C: memory usage validation with heavy load."""
        client = ChatClient(
            response_filters=[],
            context_window_size=100,
            enhanced_context=True,
        )

        # Add messages and track memory growth
        initial_stats = client._get_context_memory_usage()

        # Add 100 realistic messages
        for i in range(100):
            await client.add_message_to_context(
                f"This is a realistic chat message {i} with typical length and content for testing memory usage patterns and growth",
                f"user{i % 15}",
                is_bot=(i % 8 == 0),
            )

        final_stats = client._get_context_memory_usage()

        # Memory validation
        assert final_stats["message_count"] == 100
        assert final_stats["total_context_bytes"] > initial_stats["total_context_bytes"]
        assert (
            final_stats["total_context_bytes"] < 1024 * 1024
        )  # < 1MB for 100 messages
        assert final_stats["avg_message_size"] > 0
        assert final_stats["estimated_tokens"] > 0

    @pytest.mark.asyncio
    async def test_context_retrieval_consistency_under_load(self) -> None:
        """Test KEP-001 Increment C: context retrieval consistency with heavy usage."""
        client = ChatClient(
            response_filters=[],
            context_window_size=50,
            enhanced_context=True,
            prioritize_mentions=True,
            max_context_tokens=800,
        )

        # Add messages with known patterns
        for i in range(50):
            if i == 25:  # Add a mention in the middle
                await client.add_message_to_context(
                    "@brok what's the weather?", "alice"
                )
            else:
                await client.add_message_to_context(f"Message {i}", f"user{i % 5}")

        # Test multiple context retrievals for consistency
        context1 = client.get_context(current_sender="alice")
        context2 = client.get_context(current_sender="alice")
        context3 = client.get_context(current_sender="user1")

        # Results should be consistent
        assert context1 == context2
        assert context1 is not None
        assert context3 is not None

        # Alice's context should prioritize her messages
        assert "alice:" in context1

        # Verify context length is reasonable
        assert len(context1) < 5000  # Should be bounded by token limit

    @pytest.mark.asyncio
    async def test_memory_bounds_validation_and_enforcement(self) -> None:
        """Test KEP-001 Increment C: memory bounds validation and enforcement."""
        client = ChatClient(
            response_filters=[],
            context_window_size=20,  # Small window for easier testing
            enhanced_context=True,
        )

        # Add messages to trigger memory validation
        for i in range(20):
            await client.add_message_to_context(
                f"Message {i} " + "x" * 50,  # Moderate messages to test memory bounds
                f"user{i % 3}",
                is_bot=(i % 5 == 0),
            )

        # Test memory validation
        validation_results = client._validate_memory_bounds()

        # All validations should pass for reasonable usage
        assert validation_results["message_count_reasonable"]
        assert validation_results["token_count_reasonable"]

        # Test memory enforcement (should not raise exceptions)
        client._enforce_memory_bounds()

        # Memory usage should still be within bounds after enforcement
        post_enforcement_validation = client._validate_memory_bounds()
        assert post_enforcement_validation["within_warning_bounds"]

    @pytest.mark.asyncio
    async def test_context_performance_metrics_logging(self) -> None:
        """Test KEP-001 Increment C: performance metrics logging functionality."""
        client = ChatClient(
            response_filters=[],
            context_window_size=50,
            enhanced_context=True,
            prioritize_mentions=True,
        )

        # Add messages for testing
        for i in range(50):
            await client.add_message_to_context(f"Test message {i}", f"user{i % 5}")

        # Test context retrieval with metrics (should not raise exceptions)
        context = client.get_context_with_metrics(current_sender="user1")

        assert context is not None
        assert len(context) > 0

        # Test that metrics methods work without errors
        messages = list(client._context_messages_structured)

        # Test prioritization with metrics
        prioritized = client._prioritize_context_messages_with_metrics(
            messages, "user1"
        )
        assert len(prioritized) == len(messages)

        # Test token limiting with metrics
        limited = client._apply_token_limit_with_metrics(messages)
        assert len(limited) <= len(messages)


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


class TestMentionDetection:
    """Test cases for mention detection utility function."""

    def test_is_mention_at_symbol(self):
        """Test @botname mention detection."""
        test_cases = [
            ("@brok hello", "brok", True),
            ("hello @brok there", "brok", True),
            ("@BROK case insensitive", "brok", True),
            ("@brok", "brok", True),  # Just the mention
            ("@wrongbot hello", "brok", False),
            ("no mention here", "brok", False),
            ("", "brok", False),
        ]

        for message, bot_name, expected in test_cases:
            result = is_mention(message, bot_name)
            assert result == expected, (
                f"Failed for message: '{message}' with bot: '{bot_name}'"
            )

    def test_is_mention_colon_format(self):
        """Test botname: mention detection."""
        test_cases = [
            ("brok: hello there", "brok", True),
            ("BROK: case insensitive", "brok", True),
            ("brok: ", "brok", True),  # Just colon
            ("notbrok: hello", "brok", False),
            ("hello brok: middle", "brok", False),  # Must be at start
            ("", "brok", False),
        ]

        for message, bot_name, expected in test_cases:
            result = is_mention(message, bot_name)
            assert result == expected, (
                f"Failed for message: '{message}' with bot: '{bot_name}'"
            )

    def test_is_mention_comma_format(self):
        """Test botname, mention detection."""
        test_cases = [
            ("brok, hello there", "brok", True),
            ("BROK, case insensitive", "brok", True),
            ("brok,", "brok", True),  # Just comma
            ("notbrok, hello", "brok", False),
            ("hello brok, middle", "brok", False),  # Must be at start
            ("", "brok", False),
        ]

        for message, bot_name, expected in test_cases:
            result = is_mention(message, bot_name)
            assert result == expected, (
                f"Failed for message: '{message}' with bot: '{bot_name}'"
            )

    def test_is_mention_first_word(self):
        """Test botname as first word mention detection."""
        test_cases = [
            ("brok hello there", "brok", True),
            ("BROK hello there", "brok", True),
            ("brok", "brok", True),  # Just the name
            ("notbrok hello", "brok", False),
            ("hello brok", "brok", False),  # Must be first word
            ("", "brok", False),
        ]

        for message, bot_name, expected in test_cases:
            result = is_mention(message, bot_name)
            assert result == expected, (
                f"Failed for message: '{message}' with bot: '{bot_name}'"
            )


class TestCommandParsing:
    """Test cases for command parsing utility function."""

    def test_parse_command_exclamation(self):
        """Test !command parsing."""
        test_cases = [
            ("!help", "brok", ("help", [])),
            ("!status check now", "brok", ("status", ["check", "now"])),
            ("!HELP", "brok", ("help", [])),  # Case insensitive
            ("!", "brok", None),  # Just exclamation
            ("!  ", "brok", None),  # Just exclamation with spaces
            ("", "brok", None),
            ("no exclamation", "brok", None),
        ]

        for message, bot_name, expected in test_cases:
            result = parse_command(message, bot_name)
            if expected is None:
                assert result is None, f"Expected None for message: '{message}'"
            else:
                expected_cmd, expected_args = expected
                assert result is not None, f"Expected command for message: '{message}'"
                assert result.command == expected_cmd, (
                    f"Wrong command for message: '{message}'"
                )
                assert result.args == expected_args, (
                    f"Wrong args for message: '{message}'"
                )

    def test_parse_command_at_mention(self):
        """Test @botname command parsing."""
        test_cases = [
            ("@brok help", "brok", ("help", [])),
            ("@brok status check", "brok", ("status", ["check"])),
            ("@BROK HELP", "brok", ("help", [])),  # Case insensitive
            ("@brok", "brok", None),  # Just mention
            ("@brok  ", "brok", None),  # Just mention with spaces
            ("@wrongbot help", "brok", None),  # Wrong bot
            ("", "brok", None),
        ]

        for message, bot_name, expected in test_cases:
            result = parse_command(message, bot_name)
            if expected is None:
                assert result is None, f"Expected None for message: '{message}'"
            else:
                expected_cmd, expected_args = expected
                assert result is not None, f"Expected command for message: '{message}'"
                assert result.command == expected_cmd, (
                    f"Wrong command for message: '{message}'"
                )
                assert result.args == expected_args, (
                    f"Wrong args for message: '{message}'"
                )

    def test_parse_command_colon_format(self):
        """Test botname: command parsing."""
        test_cases = [
            ("brok: help", "brok", ("help", [])),
            ("brok: status check", "brok", ("status", ["check"])),
            ("BROK: HELP", "brok", ("help", [])),  # Case insensitive
            ("brok:", "brok", None),  # Just colon
            ("brok:  ", "brok", None),  # Just colon with spaces
            ("wrongbot: help", "brok", None),  # Wrong bot
            ("", "brok", None),
        ]

        for message, bot_name, expected in test_cases:
            result = parse_command(message, bot_name)
            if expected is None:
                assert result is None, f"Expected None for message: '{message}'"
            else:
                expected_cmd, expected_args = expected
                assert result is not None, f"Expected command for message: '{message}'"
                assert result.command == expected_cmd, (
                    f"Wrong command for message: '{message}'"
                )
                assert result.args == expected_args, (
                    f"Wrong args for message: '{message}'"
                )

    def test_parse_command_comma_format(self):
        """Test botname, command parsing."""
        test_cases = [
            ("brok, help", "brok", ("help", [])),
            ("brok, status check", "brok", ("status", ["check"])),
            ("BROK, HELP", "brok", ("help", [])),  # Case insensitive
            ("brok,", "brok", None),  # Just comma
            ("brok,  ", "brok", None),  # Just comma with spaces
            ("wrongbot, help", "brok", None),  # Wrong bot
            ("", "brok", None),
        ]

        for message, bot_name, expected in test_cases:
            result = parse_command(message, bot_name)
            if expected is None:
                assert result is None, f"Expected None for message: '{message}'"
            else:
                expected_cmd, expected_args = expected
                assert result is not None, f"Expected command for message: '{message}'"
                assert result.command == expected_cmd, (
                    f"Wrong command for message: '{message}'"
                )
                assert result.args == expected_args, (
                    f"Wrong args for message: '{message}'"
                )

    def test_parse_command_first_word(self):
        """Test botname command parsing (botname as first word)."""
        test_cases = [
            ("brok help", "brok", ("help", [])),
            ("brok status check", "brok", ("status", ["check"])),
            ("BROK HELP", "brok", ("help", [])),  # Case insensitive
            ("brok", "brok", None),  # Just bot name
            ("wrongbot help", "brok", None),  # Wrong bot
            ("", "brok", None),
        ]

        for message, bot_name, expected in test_cases:
            result = parse_command(message, bot_name)
            if expected is None:
                assert result is None, f"Expected None for message: '{message}'"
            else:
                expected_cmd, expected_args = expected
                assert result is not None, f"Expected command for message: '{message}'"
                assert result.command == expected_cmd, (
                    f"Wrong command for message: '{message}'"
                )
                assert result.args == expected_args, (
                    f"Wrong args for message: '{message}'"
                )


class TestChatStats:
    """Test cases for ChatStats."""

    def test_chat_stats_default_initialization(self):
        """Test ChatStats initialization with default values."""
        # Act
        stats = ChatStats()

        # Assert
        assert stats.last_activity == 0.0
        assert stats.messages_received == 0
        assert stats.start_time == 0.0
        assert stats.reconnections == 0

    def test_chat_stats_with_values(self):
        """Test ChatStats initialization with custom values."""
        # Act
        stats = ChatStats(
            last_activity=1234567900.0,
            messages_received=42,
            start_time=1234567890.0,
            reconnections=3,
        )

        # Assert
        assert stats.last_activity == 1234567900.0
        assert stats.messages_received == 42
        assert stats.start_time == 1234567890.0
        assert stats.reconnections == 3

    @pytest.mark.asyncio
    async def test_chat_client_updates_stats_on_message(self, chat_client: ChatClient):
        """Test that chat client updates stats when receiving messages."""
        # Arrange - Mock the wsggpy Message
        mock_user = MagicMock(spec=User)
        mock_user.nick = "testuser"

        mock_message = MagicMock(spec=Message)
        mock_message.sender = mock_user
        mock_message.message = "Hello world"
        mock_message.timestamp = time.time() * 1000  # Milliseconds

        # Get initial stats
        initial_stats = chat_client.get_chat_stats()
        initial_count = initial_stats.messages_received
        initial_activity = initial_stats.last_activity

        # Act - Simulate message reception
        chat_client._on_message(mock_message, MagicMock(spec=AsyncSession))

        # Wait for async tasks to complete
        await asyncio.sleep(0.1)

        # Assert
        updated_stats = chat_client.get_chat_stats()
        assert updated_stats.messages_received == initial_count + 1
        assert updated_stats.last_activity > initial_activity

    @pytest.mark.asyncio
    async def test_reconnection_resets_last_activity(self, chat_client: ChatClient):
        """Test that reconnection resets last_activity to prevent stale connection loops."""
        # Arrange - Set an old last_activity timestamp to simulate stale connection
        old_timestamp = time.time() - 600  # 10 minutes ago
        chat_client._chat_stats.last_activity = old_timestamp
        initial_reconnections = chat_client._chat_stats.reconnections

        # Act - Simulate successful reconnection event
        mock_session = MagicMock(spec=AsyncSession)
        chat_client._on_reconnected(None, mock_session)

        # Assert - The key fix: last_activity should be reset to current time
        stats = chat_client.get_chat_stats()
        assert stats.last_activity > old_timestamp  # Should be reset to current time
        assert (
            stats.last_activity >= time.time() - 1
        )  # Should be very recent (within 1 second)
        assert (
            stats.reconnections == initial_reconnections + 1
        )  # Should increment reconnection counter
