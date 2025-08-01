"""Configuration management for the chatbot."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
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
    llm_max_tokens: int = 100  # Reduced from 150 for more concise responses
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

    # Enhanced context settings (KEP-001) - Now enabled by default
    enhanced_context: bool = True  # Feature flag for structured context - ENABLED
    max_context_tokens: int = 500
    prioritize_mentions: bool = True
    include_bot_responses: bool = True

    # Prompt settings
    custom_system_prompt: str | None = None  # Override default system prompt

    # Connection settings
    max_reconnect_attempts: int = (
        10  # Max consecutive reconnection failures before giving up
    )
    initial_reconnect_delay: float = (
        5.0  # Initial delay between reconnection attempts (seconds)
    )
    max_reconnect_delay: float = (
        300.0  # Maximum delay between reconnection attempts (seconds)
    )
    connection_check_interval: int = (
        10  # How often to check connection status (seconds)
    )

    # wsggpy auto-reconnection settings (NEW)
    wsggpy_auto_reconnect: bool = True  # Enable wsggpy's built-in auto-reconnection
    wsggpy_reconnect_attempts: int = 5  # Number of reconnection attempts for wsggpy
    wsggpy_reconnect_delay: float = (
        2.0  # Initial delay between wsggpy reconnection attempts
    )
    wsggpy_reconnect_backoff: bool = (
        True  # Whether to use exponential backoff in wsggpy
    )

    # Tools configuration
    enable_tools: bool = True  # Whether to enable tool calling

    # Logging
    log_level: str = "INFO"
    log_chat_messages: bool = False  # Opt-in for privacy
    log_prompt_tokens: bool = False  # Opt-in for performance monitoring

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
            max_tokens = cls._parse_positive_int(
                "LLM_MAX_TOKENS", "100"
            )  # Reduced default for concise responses
            temperature = cls._parse_float_range("LLM_TEMPERATURE", "0.7", 0.0, 2.0)
            timeout_seconds = cls._parse_positive_int("LLM_TIMEOUT", "30")
            max_concurrent = cls._parse_positive_int("LLM_MAX_CONCURRENT", "2")
            context_window_size = cls._parse_positive_int("CONTEXT_WINDOW_SIZE", "10")

            # Parse connection settings
            max_reconnect_attempts = cls._parse_positive_int(
                "MAX_RECONNECT_ATTEMPTS", "10"
            )
            initial_reconnect_delay = cls._parse_float_range(
                "INITIAL_RECONNECT_DELAY", "5.0", 1.0, 60.0
            )
            max_reconnect_delay = cls._parse_float_range(
                "MAX_RECONNECT_DELAY", "300.0", 30.0, 3600.0
            )
            connection_check_interval = cls._parse_positive_int(
                "CONNECTION_CHECK_INTERVAL", "10"
            )

            # Parse wsggpy auto-reconnection settings
            wsggpy_auto_reconnect = (
                os.getenv("WSGGPY_AUTO_RECONNECT", "true").lower() == "true"
            )
            wsggpy_reconnect_attempts = cls._parse_positive_int(
                "WSGGPY_RECONNECT_ATTEMPTS", "5"
            )
            wsggpy_reconnect_delay = cls._parse_float_range(
                "WSGGPY_RECONNECT_DELAY", "2.0", 0.1, 60.0
            )
            wsggpy_reconnect_backoff = (
                os.getenv("WSGGPY_RECONNECT_BACKOFF", "true").lower() == "true"
            )

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

            # Parse ignore users list
            ignore_users_str = os.getenv("BOT_IGNORE_USERS", "")
            ignore_users = (
                [user.strip() for user in ignore_users_str.split(",") if user.strip()]
                if ignore_users_str
                else []
            )

            # Parse boolean flags
            respond_to_mentions = (
                os.getenv("BOT_RESPOND_TO_MENTIONS", "true").lower() == "true"
            )
            respond_to_commands = (
                os.getenv("BOT_RESPOND_TO_COMMANDS", "true").lower() == "true"
            )

            # Parse enhanced context settings (KEP-001) - Now defaults to enabled
            enhanced_context = os.getenv("ENHANCED_CONTEXT", "true").lower() == "true"
            max_context_tokens = cls._parse_positive_int("MAX_CONTEXT_TOKENS", "500")
            prioritize_mentions = (
                os.getenv("PRIORITIZE_MENTIONS", "true").lower() == "true"
            )
            include_bot_responses = (
                os.getenv("INCLUDE_BOT_RESPONSES", "true").lower() == "true"
            )

            # Validate enhanced context configuration (KEP-001 Increment C)
            cls._validate_enhanced_context_config(
                enhanced_context,
                max_context_tokens,
                context_window_size,
                prioritize_mentions,
                include_bot_responses,
            )

            # Parse prompt configuration
            custom_system_prompt = os.getenv("CUSTOM_SYSTEM_PROMPT")

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
                ignore_users=ignore_users,
                context_window_size=context_window_size,
                enhanced_context=enhanced_context,
                max_context_tokens=max_context_tokens,
                prioritize_mentions=prioritize_mentions,
                include_bot_responses=include_bot_responses,
                custom_system_prompt=custom_system_prompt,
                max_reconnect_attempts=max_reconnect_attempts,
                initial_reconnect_delay=initial_reconnect_delay,
                max_reconnect_delay=max_reconnect_delay,
                connection_check_interval=connection_check_interval,
                wsggpy_auto_reconnect=wsggpy_auto_reconnect,
                wsggpy_reconnect_attempts=wsggpy_reconnect_attempts,
                wsggpy_reconnect_delay=wsggpy_reconnect_delay,
                wsggpy_reconnect_backoff=wsggpy_reconnect_backoff,
                log_level=log_level,
                log_chat_messages=os.getenv("LOG_CHAT", "false").lower() == "true",
                log_prompt_tokens=os.getenv("LOG_PROMPT_TOKENS", "false").lower()
                == "true",
                enable_tools=os.getenv("ENABLE_TOOLS", "true").lower() == "true",
            )
        except ValueError as e:
            raise ConfigurationError(f"Invalid configuration value: {e}") from e

    @classmethod
    def _validate_enhanced_context_config(
        cls,
        enhanced_context: bool,
        max_context_tokens: int,
        context_window_size: int,
        prioritize_mentions: bool,
        include_bot_responses: bool,
    ) -> None:
        """Validate enhanced context configuration for KEP-001 Increment C.

        Args:
            enhanced_context: Whether enhanced context is enabled
            max_context_tokens: Maximum tokens for context
            context_window_size: Window size for message history
            prioritize_mentions: Whether to prioritize mentions
            include_bot_responses: Whether to include bot responses

        Raises:
            ConfigurationError: When configuration values are invalid
        """
        # Validate token limits
        if max_context_tokens < 10:
            raise ConfigurationError(
                f"MAX_CONTEXT_TOKENS must be at least 10, got: {max_context_tokens}"
            )

        if max_context_tokens > 10000:
            raise ConfigurationError(
                f"MAX_CONTEXT_TOKENS should not exceed 10000 for performance, got: {max_context_tokens}"
            )

        # Validate context window size bounds
        if context_window_size < 1:
            raise ConfigurationError(
                f"CONTEXT_WINDOW_SIZE must be at least 1, got: {context_window_size}"
            )

        if context_window_size > 1000:
            raise ConfigurationError(
                f"CONTEXT_WINDOW_SIZE should not exceed 1000 for memory efficiency, got: {context_window_size}"
            )

        # Warn about potential performance issues if enhanced context is disabled
        # but large window size is configured
        if not enhanced_context and context_window_size > 50:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Enhanced context disabled but large window size ({context_window_size}) configured. "
                "Consider enabling enhanced context for better performance with large windows."
            )

        # Validate logical configuration combinations
        if enhanced_context and not include_bot_responses and not prioritize_mentions:
            logger = logging.getLogger(__name__)
            logger.warning(
                "Enhanced context enabled but both bot responses and mention prioritization disabled. "
                "This may result in limited context utility."
            )

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
