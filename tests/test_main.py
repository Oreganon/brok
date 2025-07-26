"""Tests for brok main module."""

from __future__ import annotations

import pytest

from brok.main import parse_args


class TestParseArgs:
    """Test cases for parse_args function."""

    def test_parse_args_default_values(self):
        """Test that parse_args returns expected default values."""
        # Arrange & Act
        args = parse_args([])

        # Assert
        assert args.dev is False
        assert args.jwt is None
        assert args.llm_url is None
        assert args.llm_provider is None
        assert args.log_level is None

    def test_parse_args_dev_flag(self):
        """Test that --dev flag is correctly parsed."""
        # Arrange & Act
        args = parse_args(["--dev"])

        # Assert
        assert args.dev is True

    def test_parse_args_jwt_token(self):
        """Test that --jwt argument is correctly parsed."""
        # Arrange
        test_jwt = "test.jwt.token"

        # Act
        args = parse_args(["--jwt", test_jwt])

        # Assert
        assert args.jwt == test_jwt

    def test_parse_args_llm_url(self):
        """Test that --llm-url argument is correctly parsed."""
        # Arrange
        test_url = "http://localhost:8080"

        # Act
        args = parse_args(["--llm-url", test_url])

        # Assert
        assert args.llm_url == test_url

    def test_parse_args_llm_provider_ollama(self):
        """Test that --llm-provider ollama is correctly parsed."""
        # Arrange & Act
        args = parse_args(["--llm-provider", "ollama"])

        # Assert
        assert args.llm_provider == "ollama"

    def test_parse_args_llm_provider_llamacpp(self):
        """Test that --llm-provider llamacpp is correctly parsed."""
        # Arrange & Act
        args = parse_args(["--llm-provider", "llamacpp"])

        # Assert
        assert args.llm_provider == "llamacpp"

    def test_parse_args_llm_provider_invalid_choice(self):
        """Test that invalid --llm-provider choice raises SystemExit."""
        # Arrange & Act & Assert
        with pytest.raises(SystemExit):
            parse_args(["--llm-provider", "invalid"])

    def test_parse_args_log_level_debug(self):
        """Test that --log-level DEBUG is correctly parsed."""
        # Arrange & Act
        args = parse_args(["--log-level", "DEBUG"])

        # Assert
        assert args.log_level == "DEBUG"

    def test_parse_args_log_level_info(self):
        """Test that --log-level INFO is correctly parsed."""
        # Arrange & Act
        args = parse_args(["--log-level", "INFO"])

        # Assert
        assert args.log_level == "INFO"

    def test_parse_args_log_level_warning(self):
        """Test that --log-level WARNING is correctly parsed."""
        # Arrange & Act
        args = parse_args(["--log-level", "WARNING"])

        # Assert
        assert args.log_level == "WARNING"

    def test_parse_args_log_level_error(self):
        """Test that --log-level ERROR is correctly parsed."""
        # Arrange & Act
        args = parse_args(["--log-level", "ERROR"])

        # Assert
        assert args.log_level == "ERROR"

    def test_parse_args_log_level_critical(self):
        """Test that --log-level CRITICAL is correctly parsed."""
        # Arrange & Act
        args = parse_args(["--log-level", "CRITICAL"])

        # Assert
        assert args.log_level == "CRITICAL"

    def test_parse_args_log_level_invalid_choice(self):
        """Test that invalid --log-level choice raises SystemExit."""
        # Arrange & Act & Assert
        with pytest.raises(SystemExit):
            parse_args(["--log-level", "INVALID"])

    def test_parse_args_log_level_case_sensitive(self):
        """Test that --log-level is case sensitive."""
        # Arrange & Act & Assert
        with pytest.raises(SystemExit):
            parse_args(["--log-level", "debug"])  # lowercase should fail

    def test_parse_args_combined_flags(self):
        """Test parsing multiple arguments together."""
        # Arrange
        test_jwt = "test.jwt.token"
        test_url = "http://localhost:8080"

        # Act
        args = parse_args(
            [
                "--dev",
                "--jwt",
                test_jwt,
                "--llm-url",
                test_url,
                "--llm-provider",
                "llamacpp",
                "--log-level",
                "DEBUG",
            ]
        )

        # Assert
        assert args.dev is True
        assert args.jwt == test_jwt
        assert args.llm_url == test_url
        assert args.llm_provider == "llamacpp"
        assert args.log_level == "DEBUG"

    def test_parse_args_help_flag_exits(self):
        """Test that --help flag causes SystemExit."""
        # Arrange & Act & Assert
        with pytest.raises(SystemExit):
            parse_args(["--help"])
