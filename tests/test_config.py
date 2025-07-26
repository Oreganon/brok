"""Tests for brok configuration module."""

from __future__ import annotations

import pytest

from brok.config import BotConfig
from brok.exceptions import ConfigurationError


class TestBotConfig:
    """Test cases for BotConfig class."""

    def test_default_config_creation(self):
        """Test creating BotConfig with default values."""
        # Arrange & Act
        config = BotConfig()

        # Assert
        assert config.chat_environment == "production"
        assert config.jwt_token is None
        assert config.llm_provider == "ollama"
        assert config.llm_model == "llama3.2:3b"
        assert config.llm_base_url == "http://localhost:11434"
        assert config.llm_max_tokens == 150
        assert config.llm_temperature == 0.7
        assert config.llm_timeout_seconds == 30
        assert config.llm_max_concurrent_requests == 2
        assert config.respond_to_keywords == ["!bot", "!ask"]
        assert config.ignore_users == []
        assert config.context_window_size == 10
        assert config.log_level == "INFO"
        assert config.log_chat_messages is False

    def test_from_env_with_defaults(self, monkeypatch):
        """Test loading config from environment with all defaults."""
        # Arrange - clear any existing env vars
        for key in [
            "CHAT_ENV",
            "STRIMS_JWT",
            "LLM_PROVIDER",
            "LLM_MODEL",
            "LLM_BASE_URL",
            "LLM_MAX_TOKENS",
            "LLM_TEMPERATURE",
            "LLM_TIMEOUT",
            "LLM_MAX_CONCURRENT",
            "BOT_KEYWORDS",
            "CONTEXT_WINDOW_SIZE",
            "LOG_LEVEL",
            "LOG_CHAT",
        ]:
            monkeypatch.delenv(key, raising=False)

        # Act
        config = BotConfig.from_env()

        # Assert
        assert config.chat_environment == "production"
        assert config.llm_provider == "ollama"
        assert config.llm_temperature == 0.7
        assert config.respond_to_keywords == ["!bot", "!ask"]

    def test_from_env_with_custom_values(self, monkeypatch):
        """Test loading config from environment with custom values."""
        # Arrange
        monkeypatch.setenv("CHAT_ENV", "dev")
        monkeypatch.setenv("STRIMS_JWT", "test-token")
        monkeypatch.setenv("LLM_PROVIDER", "llamacpp")
        monkeypatch.setenv("LLM_MODEL", "custom-model")
        monkeypatch.setenv("LLM_MAX_TOKENS", "200")
        monkeypatch.setenv("LLM_TEMPERATURE", "0.9")
        monkeypatch.setenv("BOT_KEYWORDS", "!test,!custom")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("LOG_CHAT", "true")

        # Act
        config = BotConfig.from_env()

        # Assert
        assert config.chat_environment == "dev"
        assert config.jwt_token == "test-token"
        assert config.llm_provider == "llamacpp"
        assert config.llm_model == "custom-model"
        assert config.llm_max_tokens == 200
        assert config.llm_temperature == 0.9
        assert config.respond_to_keywords == ["!test", "!custom"]
        assert config.log_level == "DEBUG"
        assert config.log_chat_messages is True

    @pytest.mark.parametrize(
        "env_value,expected_error",
        [
            ("invalid", "CHAT_ENV must be 'production' or 'dev'"),
            ("test", "CHAT_ENV must be 'production' or 'dev'"),
            ("", "CHAT_ENV must be 'production' or 'dev'"),
        ],
    )
    def test_from_env_invalid_chat_environment(
        self, monkeypatch, env_value, expected_error
    ):
        """Test validation of CHAT_ENV environment variable."""
        # Arrange
        monkeypatch.setenv("CHAT_ENV", env_value)

        # Act & Assert
        with pytest.raises(ConfigurationError, match=expected_error):
            BotConfig.from_env()

    @pytest.mark.parametrize(
        "env_value,expected_error",
        [
            ("invalid", "LLM_PROVIDER must be 'ollama' or 'llamacpp'"),
            ("openai", "LLM_PROVIDER must be 'ollama' or 'llamacpp'"),
            ("", "LLM_PROVIDER must be 'ollama' or 'llamacpp'"),
        ],
    )
    def test_from_env_invalid_llm_provider(
        self, monkeypatch, env_value, expected_error
    ):
        """Test validation of LLM_PROVIDER environment variable."""
        # Arrange
        monkeypatch.setenv("LLM_PROVIDER", env_value)

        # Act & Assert
        with pytest.raises(ConfigurationError, match=expected_error):
            BotConfig.from_env()

    @pytest.mark.parametrize(
        "env_value,expected_error",
        [
            ("not_a_number", "Invalid configuration value.*Invalid LLM_MAX_TOKENS"),
            ("0", "Invalid configuration value.*Invalid LLM_MAX_TOKENS"),
            ("-5", "Invalid configuration value.*Invalid LLM_MAX_TOKENS"),
            ("", "Invalid configuration value.*Invalid LLM_MAX_TOKENS"),
        ],
    )
    def test_from_env_invalid_max_tokens(self, monkeypatch, env_value, expected_error):
        """Test validation of LLM_MAX_TOKENS environment variable."""
        # Arrange
        monkeypatch.setenv("LLM_MAX_TOKENS", env_value)

        # Act & Assert
        with pytest.raises(ConfigurationError, match=expected_error):
            BotConfig.from_env()

    @pytest.mark.parametrize(
        "env_value,expected_error",
        [
            ("not_a_number", "Invalid configuration value.*Invalid LLM_TEMPERATURE"),
            ("-0.1", "Invalid configuration value.*Invalid LLM_TEMPERATURE"),
            ("2.1", "Invalid configuration value.*Invalid LLM_TEMPERATURE"),
            ("", "Invalid configuration value.*Invalid LLM_TEMPERATURE"),
        ],
    )
    def test_from_env_invalid_temperature(self, monkeypatch, env_value, expected_error):
        """Test validation of LLM_TEMPERATURE environment variable."""
        # Arrange
        monkeypatch.setenv("LLM_TEMPERATURE", env_value)

        # Act & Assert
        with pytest.raises(ConfigurationError, match=expected_error):
            BotConfig.from_env()

    @pytest.mark.parametrize(
        "env_value,expected_error",
        [
            ("INVALID", "LOG_LEVEL must be valid logging level"),
            ("trace", "LOG_LEVEL must be valid logging level"),
            ("", "LOG_LEVEL must be valid logging level"),
        ],
    )
    def test_from_env_invalid_log_level(self, monkeypatch, env_value, expected_error):
        """Test validation of LOG_LEVEL environment variable."""
        # Arrange
        monkeypatch.setenv("LOG_LEVEL", env_value)

        # Act & Assert
        with pytest.raises(ConfigurationError, match=expected_error):
            BotConfig.from_env()

    def test_from_env_empty_keywords(self, monkeypatch):
        """Test validation of empty BOT_KEYWORDS."""
        # Arrange
        monkeypatch.setenv("BOT_KEYWORDS", "")

        # Act & Assert
        with pytest.raises(ConfigurationError, match="BOT_KEYWORDS cannot be empty"):
            BotConfig.from_env()

    def test_from_env_keywords_whitespace_handling(self, monkeypatch):
        """Test that keywords are properly trimmed and filtered."""
        # Arrange
        monkeypatch.setenv("BOT_KEYWORDS", " !test , !bot ,  , !ask ")

        # Act
        config = BotConfig.from_env()

        # Assert
        assert config.respond_to_keywords == ["!test", "!bot", "!ask"]

    @pytest.mark.parametrize(
        "log_chat_value,expected",
        [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("", False),
            ("invalid", False),
        ],
    )
    def test_from_env_log_chat_parsing(self, monkeypatch, log_chat_value, expected):
        """Test parsing of LOG_CHAT environment variable."""
        # Arrange
        monkeypatch.setenv("LOG_CHAT", log_chat_value)

        # Act
        config = BotConfig.from_env()

        # Assert
        assert config.log_chat_messages == expected
