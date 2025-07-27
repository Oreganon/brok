#!/usr/bin/env python3
"""
Test script for wsggpy auto-reconnection functionality.

This script demonstrates the new reconnection features and can be used
to test the integration with the updated ChatClient.
"""

import asyncio
import logging
import os
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from brok.chat import ChatClient, create_default_filters  # noqa: E402
from brok.config import BotConfig  # noqa: E402


async def test_reconnection() -> None:
    """Test the new wsggpy reconnection functionality."""

    # Set up logging to see reconnection events
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("🧪 Starting wsggpy reconnection test...")

    # Create test configuration
    config = BotConfig(
        chat_environment="dev",  # Use dev environment for testing
        jwt_token=os.getenv("STRIMS_JWT"),  # Optional authentication
        bot_name="test-brok",
        respond_to_keywords=["!test"],
        wsggpy_auto_reconnect=True,
        wsggpy_reconnect_attempts=3,
        wsggpy_reconnect_delay=1.0,  # Fast reconnection for testing
        connection_check_interval=5,  # More frequent checks
    )

    # Create chat client with reconnection enabled
    filters = create_default_filters(config.respond_to_keywords)
    chat_client = ChatClient(
        response_filters=filters,
        context_window_size=5,
        bot_name=config.bot_name,
        wsggpy_auto_reconnect=config.wsggpy_auto_reconnect,
        wsggpy_reconnect_attempts=config.wsggpy_reconnect_attempts,
        wsggpy_reconnect_delay=config.wsggpy_reconnect_delay,
    )

    try:
        # Test 1: Normal connection
        logger.info("✅ Test 1: Normal connection")
        await chat_client.connect(
            jwt_token=config.jwt_token, environment=config.chat_environment
        )

        # Wait a bit to see initial connection status
        await asyncio.sleep(2)

        # Test 2: Check connection status
        logger.info("✅ Test 2: Connection status check")
        logger.info(f"Connected: {chat_client.is_connected()}")
        logger.info(f"Reconnecting: {chat_client.is_reconnecting()}")

        conn_info = chat_client.get_connection_info()
        logger.info(f"Connection info: {conn_info}")

        # Test 3: Manual reconnection
        logger.info("✅ Test 3: Manual reconnection test")
        logger.info("Forcing reconnection...")
        await chat_client.force_reconnect()

        # Wait for reconnection to complete
        await asyncio.sleep(5)

        # Check status after reconnection
        logger.info(f"After reconnection - Connected: {chat_client.is_connected()}")
        logger.info(f"Connection info: {chat_client.get_connection_info()}")

        # Test 4: Monitor for a while to see automatic behavior
        logger.info("✅ Test 4: Monitoring for 30 seconds...")
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < 30:
            await asyncio.sleep(5)
            logger.info(
                f"Status check - Connected: {chat_client.is_connected()}, "
                f"Reconnecting: {chat_client.is_reconnecting()}"
            )

        logger.info("🎉 Reconnection test completed successfully!")

    except KeyboardInterrupt:
        logger.info("⏹️  Test interrupted by user")
    except Exception:
        logger.exception("❌ Test failed with error")
        raise
    finally:
        # Clean up
        await chat_client.disconnect()
        logger.info("🧹 Test cleanup completed")


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_reconnection())
