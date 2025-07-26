"""Configuration management for the chatbot."""

from __future__ import annotations

from dataclasses import dataclass, field
import os

from brok.exceptions import ConfigurationError


@dataclass
class BotConfig:
    """Simplified configuration for the chatbot."""

    # Chat settings
    chat_environment: str = "production"  # or "dev"
    jwt_token: str | None = None

    # LLM settings
    llm_provider: str = "ollama"  # "ollama", "llamacpp"
    llm_model: str = "llama3.2:3b"
    llm_base_url: str = "http://localhost:11434"
    llm_max_tokens: int = 150
    llm_temperature: float = 0.7
    llm_timeout_seconds: int = 30
    llm_max_concurrent_requests: int = 2

    # Bot behavior
    bot_name: str = "brok"  # Bot name for mention detection
    respond_to_keywords: list[str] = field(default_factory=lambda: ["!bot", "!ask"])
    respond_to_mentions: bool = True  # Respond when bot is mentioned
    respond_to_commands: bool = True  # Parse and respond to commands
    ignore_users: list[str] = field(default_factory=list)
    context_window_size: int = 10

    # Logging
    log_level: str = "INFO"
    log_chat_messages: bool = False  # Opt-in for privacy

    @classmethod
    def from_env(cls) -> BotConfig:
        """Load configuration from environment variables with sensible defaults.

        Returns:
            BotConfig: Validated configuration instance

        Raises:
            ConfigurationError: When environment variables contain invalid values

        Example:
            >>> config = BotConfig.from_env()
            >>> assert config.llm_provider == "ollama"
            >>> assert 0.0 <= config.llm_temperature <= 2.0
        """
        try:
            # Parse numeric values with validation
            max_tokens = cls._parse_positive_int("LLM_MAX_TOKENS", "150")
            temperature = cls._parse_float_range("LLM_TEMPERATURE", "0.7", 0.0, 2.0)
            timeout_seconds = cls._parse_positive_int("LLM_TIMEOUT", "30")
            max_concurrent = cls._parse_positive_int("LLM_MAX_CONCURRENT", "2")
            context_window_size = cls._parse_positive_int("CONTEXT_WINDOW_SIZE", "10")

            # Validate environment
            chat_env = os.getenv("CHAT_ENV", "production")
            if chat_env not in ("production", "dev"):
                raise ConfigurationError(
                    f"CHAT_ENV must be 'production' or 'dev', got: {chat_env}"
                )

            # Validate LLM provider
            llm_provider = os.getenv("LLM_PROVIDER", "ollama")
            if llm_provider not in ("ollama", "llamacpp"):
                raise ConfigurationError(
                    f"LLM_PROVIDER must be 'ollama' or 'llamacpp', got: {llm_provider}"
                )

            # Validate log level
            log_level = os.getenv("LOG_LEVEL", "INFO").upper()
            if log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
                raise ConfigurationError(
                    f"LOG_LEVEL must be valid logging level, got: {log_level}"
                )

            # Parse keywords list
            keywords_str = os.getenv("BOT_KEYWORDS", "!bot,!ask")
            keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()]
            if not keywords:
                raise ConfigurationError("BOT_KEYWORDS cannot be empty")

            # Parse bot name
            bot_name = os.getenv("BOT_NAME", "brok").strip()
            if not bot_name:
                raise ConfigurationError("BOT_NAME cannot be empty")

            # Parse boolean flags
            respond_to_mentions = (
                os.getenv("BOT_RESPOND_TO_MENTIONS", "true").lower() == "true"
            )
            respond_to_commands = (
                os.getenv("BOT_RESPOND_TO_COMMANDS", "true").lower() == "true"
            )

            return cls(
                chat_environment=chat_env,
                jwt_token=os.getenv("STRIMS_JWT"),
                llm_provider=llm_provider,
                llm_model=os.getenv("LLM_MODEL", "llama3.2:3b"),
                llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434"),
                llm_max_tokens=max_tokens,
                llm_temperature=temperature,
                llm_timeout_seconds=timeout_seconds,
                llm_max_concurrent_requests=max_concurrent,
                bot_name=bot_name,
                respond_to_keywords=keywords,
                respond_to_mentions=respond_to_mentions,
                respond_to_commands=respond_to_commands,
                context_window_size=context_window_size,
                log_level=log_level,
                log_chat_messages=os.getenv("LOG_CHAT", "false").lower() == "true",
            )
        except ValueError as e:
            raise ConfigurationError(f"Invalid configuration value: {e}") from e

    @staticmethod
    def _parse_positive_int(env_var: str, default: str) -> int:
        """Parse environment variable as positive integer."""
        value_str = os.getenv(env_var, default)
        try:
            value = int(value_str)
            if value <= 0:
                raise ValueError(f"{env_var} must be positive, got: {value}")
            return value
        except ValueError as e:
            raise ValueError(f"Invalid {env_var}: {value_str}") from e

    @staticmethod
    def _parse_float_range(
        env_var: str, default: str, min_val: float, max_val: float
    ) -> float:
        """Parse environment variable as float within specified range."""
        value_str = os.getenv(env_var, default)
        try:
            value = float(value_str)
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"{env_var} must be between {min_val} and {max_val}, got: {value}"
                )
            return value
        except ValueError as e:
            raise ValueError(f"Invalid {env_var}: {value_str}") from e
