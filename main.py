"""Main module for brok application."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

from brok import __version__
from brok.bot import ChatBot
from brok.chat import ChatClient, create_default_filters
from brok.config import BotConfig
from brok.exceptions import ConfigurationError
from brok.llm.base import LLMConfig
from brok.llm.ollama import OllamaProvider


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Brok - AI chatbot for strims.gg chat integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start bot in production (anonymous)
  python main.py --dev              # Start bot in development chat
  python main.py --jwt TOKEN        # Start bot with authentication
  python main.py --dev --jwt TOKEN  # Start bot in dev with auth

Environment Variables:
  STRIMS_JWT          JWT token for authentication
  LLM_PROVIDER        LLM provider (default: ollama)
  LLM_MODEL           Model name (default: llama3.2:3b)
  LLM_BASE_URL        LLM API URL (default: http://localhost:11434)
  BOT_KEYWORDS        Trigger keywords (default: !bot,!ask)
  LOG_LEVEL           Logging level (default: INFO)
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

    return parser.parse_args()


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
        )

        # Create LLM provider
        if config.llm_provider == "ollama":
            llm_config = LLMConfig(
                model_name=config.llm_model,
                max_tokens=config.llm_max_tokens,
                temperature=config.llm_temperature,
                timeout_seconds=config.llm_timeout_seconds,
            )
            llm_provider = OllamaProvider(
                base_url=config.llm_base_url,
                model=config.llm_model,
                config=llm_config,
            )
        else:
            raise ConfigurationError(f"Unsupported LLM provider: {config.llm_provider}")

        # Create and start the bot
        bot = ChatBot(
            config=config,
            chat_client=chat_client,
            llm_provider=llm_provider,
        )

        await bot.start()

    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}")
        return
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logging.getLogger(__name__).exception("Unexpected error in main")


def setup_logging(level: str, log_chat_messages: bool) -> None:
    """Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_chat_messages: Whether to log chat messages
    """
    import sys

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


if __name__ == "__main__":
    asyncio.run(main())
