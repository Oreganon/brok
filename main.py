"""Main module for brok application."""

from __future__ import annotations

import argparse
import asyncio
import os

from wsggpy import AsyncSession, ChatEnvironment, Message, RoomAction


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Connect to strims.gg chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Connect to production (anonymous)
  python main.py --dev              # Connect to development chat
  python main.py --jwt TOKEN        # Connect with specific JWT token
  python main.py --dev --jwt TOKEN  # Connect to dev with authentication
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
    """Main entry point for the brok application."""
    args = parse_args()

    # Determine environment
    environment = ChatEnvironment.DEV if args.dev else ChatEnvironment.PRODUCTION
    env_name = "development" if args.dev else "production"

    print(f"Connecting to strims.gg {env_name} chat...")

    # Load JWT token (command line flag overrides environment variable)
    jwt_token = args.jwt or os.getenv("STRIMS_JWT")

    # Create async session
    session = AsyncSession(
        login_key=jwt_token,  # Will be None if not set (anonymous mode)
        url=environment,
    )

    if jwt_token:
        print(f"🔑 Using authenticated connection to {env_name}")
    else:
        print(f"👤 Using anonymous connection to {env_name}")
        print(
            "   (use --jwt TOKEN or set STRIMS_JWT environment variable to authenticate)"
        )

    # Add message handler to print incoming messages
    @session.add_message_handler
    def on_message(message: Message, _session: AsyncSession) -> None:
        print(f"[{message.sender.nick}]: {message.message}")

    # Add join/quit handlers
    @session.add_join_handler
    def on_join(event: RoomAction, _session: AsyncSession) -> None:
        print(f"👋 {event.user.nick} joined the chat")

    @session.add_quit_handler
    def on_quit(event: RoomAction, _session: AsyncSession) -> None:
        print(f"👋 {event.user.nick} left the chat")

    try:
        # Connect to chat
        await session.open()
        print(f"✅ Connected to {env_name} chat! Listening for messages...")
        if jwt_token:
            print("💬 You can send messages since you're authenticated")
        print("Press Ctrl+C to disconnect")

        # Keep the connection alive and listen for messages
        while session.is_connected():
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Disconnecting...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if session.is_connected():
            await session.close()
        print("👋 Disconnected from chat")


if __name__ == "__main__":
    asyncio.run(main())
