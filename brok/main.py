"""Main entry point for the brok chatbot."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from brok import __version__
from brok.bot import ChatBot
from brok.chat import ChatClient, create_default_filters
from brok.config import BotConfig
from brok.exceptions import ConfigurationError
from brok.llm.base import LLMConfig, LLMProvider
from brok.llm.llamacpp import LlamaCppProvider
from brok.llm.ollama import OllamaProvider
from brok.prompts import (
    PromptTemplate,
    XMLPromptTemplate,
    create_custom_lightweight_xml_template,
    create_custom_template,
    get_optimal_xml_template,
    get_prompt_template,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        args: Optional list of arguments to parse. If None, uses sys.argv.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Brok - AI chatbot for strims.gg chat integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start bot in production (anonymous)
  python main.py --dev              # Start bot in development chat
  python main.py --jwt TOKEN        # Start bot with authentication
  python main.py --dev --jwt TOKEN  # Start bot in dev with auth

  # Using different LLM providers:
  LLM_PROVIDER=ollama python main.py               # Use Ollama (default)
  LLM_PROVIDER=llamacpp python main.py             # Use LlamaCpp HTTP server
  python main.py --llm-provider llamacpp           # Use LlamaCpp via CLI flag
  python main.py --llm-url http://localhost:8080   # Override LLM API URL
  python main.py --llm-provider llamacpp --llm-url http://localhost:8080  # Combined

  # Logging configuration:
  python main.py --log-level DEBUG  # Enable debug logging
  python main.py --log-level ERROR  # Only show errors

Environment Variables:
  STRIMS_JWT                 JWT token for authentication
  LLM_PROVIDER               LLM provider: ollama, llamacpp (default: ollama)
  LLM_MODEL                  Model name (default: llama3.2:3b)
  LLM_BASE_URL               LLM API URL (default: http://localhost:11434 for ollama, http://localhost:8080 for llamacpp)
  BOT_NAME                   Bot name for mentions (default: brok)
  BOT_KEYWORDS               Trigger keywords (default: !bot,!ask)
  BOT_RESPOND_TO_MENTIONS    Respond to mentions (default: true)
  BOT_RESPOND_TO_COMMANDS    Parse and respond to commands (default: true)
  LOG_LEVEL                  Logging level (default: INFO)
        """,
    )

    parser.add_argument(
        "--dev",
        action="store_true",
        help="Connect to development chat (chat2.strims.gg) instead of production",
    )

    parser.add_argument(
        "--jwt",
        type=str,
        help="JWT token for authentication (overrides STRIMS_JWT environment variable)",
    )

    parser.add_argument(
        "--llm-url",
        type=str,
        help="LLM API base URL (overrides LLM_BASE_URL environment variable)",
    )

    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=["ollama", "llamacpp"],
        help="LLM provider to use (overrides LLM_PROVIDER environment variable)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (overrides LOG_LEVEL environment variable)",
    )

    return parser.parse_args(args)


async def main() -> None:
    """Main entry point for the brok chatbot application."""
    args = parse_args()

    try:
        # Load configuration from environment
        config = BotConfig.from_env()

        # Override config with command line arguments
        if args.dev:
            config.chat_environment = "dev"
        if args.jwt:
            config.jwt_token = args.jwt
        if args.llm_url:
            config.llm_base_url = args.llm_url
        if args.llm_provider:
            config.llm_provider = args.llm_provider
        if args.log_level:
            config.log_level = args.log_level

        # Set up logging
        setup_logging(config.log_level, config.log_chat_messages)

        logger = logging.getLogger(__name__)
        logger.info(f"Starting brok chatbot v{__version__}")
        logger.info(f"Environment: {config.chat_environment}")
        logger.info(f"LLM Provider: {config.llm_provider} ({config.llm_model})")

        # Create message filters
        filters = create_default_filters(config.respond_to_keywords)

        # Create chat client
        chat_client = ChatClient(
            response_filters=filters,
            context_window_size=config.context_window_size,
            bot_name=config.bot_name,
            respond_to_mentions=config.respond_to_mentions,
            respond_to_commands=config.respond_to_commands,
            ignore_users=config.ignore_users,
            enhanced_context=config.enhanced_context,
            max_context_tokens=config.max_context_tokens,
            prioritize_mentions=config.prioritize_mentions,
            include_bot_responses=config.include_bot_responses,
            # wsggpy auto-reconnection settings
            wsggpy_auto_reconnect=config.wsggpy_auto_reconnect,
            wsggpy_reconnect_attempts=config.wsggpy_reconnect_attempts,
            wsggpy_reconnect_delay=config.wsggpy_reconnect_delay,
            wsggpy_reconnect_backoff=config.wsggpy_reconnect_backoff,
        )

        # Create prompt template based on configuration (KEP-002 Increment B & D)
        prompt_template: PromptTemplate | XMLPromptTemplate
        if config.xml_prompt_formatting:
            # Use optimal XML template (automatically selects lightweight for 2B models)
            if config.prompt_style == "custom":
                if config.custom_system_prompt is None:
                    raise ValueError(
                        "custom_system_prompt must be provided when using custom prompt style"
                    )
                prompt_template = create_custom_lightweight_xml_template(
                    config.custom_system_prompt
                )
            else:
                prompt_template = get_optimal_xml_template(
                    config.prompt_style,
                    config.llm_model,  # Automatic 2B model detection
                )
        # Use regular PromptTemplate for backward compatibility
        elif config.prompt_style == "custom":
            if config.custom_system_prompt is None:
                raise ValueError(
                    "custom_system_prompt must be provided when using custom prompt style"
                )
            prompt_template = create_custom_template(config.custom_system_prompt)
        else:
            prompt_template = get_prompt_template(config.prompt_style)

        # Create LLM provider
        llm_config = LLMConfig(
            model_name=config.llm_model,
            max_tokens=config.llm_max_tokens,
            temperature=config.llm_temperature,
            timeout_seconds=config.llm_timeout_seconds,
            log_prompt_tokens=config.log_prompt_tokens,  # Enable prompt logging if configured
        )

        llm_provider: LLMProvider
        if config.llm_provider == "ollama":
            llm_provider = OllamaProvider(
                base_url=config.llm_base_url,
                model=config.llm_model,
                config=llm_config,
                prompt_template=prompt_template,
            )
        elif config.llm_provider == "llamacpp":
            llm_provider = LlamaCppProvider(
                base_url=config.llm_base_url,
                model=config.llm_model,
                config=llm_config,
                prompt_template=prompt_template,
            )
        else:
            raise ConfigurationError(
                f"Unsupported LLM provider: {config.llm_provider}. Supported providers: ollama, llamacpp"
            )

        # Create and start the bot
        bot = ChatBot(
            config=config,
            chat_client=chat_client,
            llm_provider=llm_provider,
        )

        await bot.start()

    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        return
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        logging.getLogger(__name__).exception("Unexpected error in main")
    finally:
        # Ensure LLM provider cleanup even if startup fails
        if "llm_provider" in locals() and hasattr(llm_provider, "close"):
            try:
                await llm_provider.close()
            except Exception as cleanup_error:
                logging.getLogger(__name__).warning(
                    f"Error during LLM provider cleanup: {cleanup_error}"
                )


def setup_logging(level: str, log_chat_messages: bool) -> None:
    """Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_chat_messages: Whether to log chat messages
    """
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    # Configure library loggers
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("wsggpy").setLevel(logging.INFO)

    # Optionally silence chat message logging for privacy
    if not log_chat_messages:
        logging.getLogger("brok.chat").setLevel(logging.WARNING)


def main_sync() -> None:
    """Synchronous entry point for console script."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
