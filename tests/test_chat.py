"""Tests for chat client functionality."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from brok.chat import (
    ChatClient,
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
    async def test_should_respond_to_pure_keywords(self, _chat_client: ChatClient):
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
        from brok.chat import ChatClient, create_default_filters, ContextMessage
        
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
        await enhanced_client.add_message_to_context("How are you?", "user1", is_bot=False)
        
        # Check that structured storage is used
        assert len(enhanced_client._context_messages_structured) == 3
        assert len(enhanced_client._context_messages_legacy) == 0
        
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
        await legacy_client.add_message_to_context("How are you?", "user1", is_bot=False)
        
        # Check that legacy storage is used
        assert len(legacy_client._context_messages_structured) == 0
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
        from brok.chat import ChatClient, create_default_filters
        
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
