"""DateTime tool for getting current date and time information."""

from __future__ import annotations

from datetime import datetime
import logging
from typing import ClassVar

try:
    import zoneinfo
except ImportError:
    zoneinfo = None

from brok.tools.base import BaseTool, ToolExecutionResult

logger = logging.getLogger(__name__)


class DateTimeTool(BaseTool):
    """Tool for getting current date and time information.

    Provides current date/time in various formats with optional timezone support.
    Can format output in different styles for user convenience.

    Example:
        >>> tool = DateTimeTool()
        >>> result = await tool.execute(format="iso")
        >>> print(result.data)  # "Current time: 2025-01-23T15:30:45"
    """

    name: ClassVar[str] = "datetime"
    description: ClassVar[str] = "Get current date and time information"
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "enum": ["iso", "readable", "date", "time", "timestamp"],
                "description": "Output format: 'iso' (ISO 8601), 'readable' (human-friendly), 'date' (date only), 'time' (time only), 'timestamp' (Unix timestamp)",
                "default": "readable",
            },
            "timezone": {
                "type": "string",
                "description": "Timezone name (e.g., 'UTC', 'US/Eastern', 'Europe/London'). Defaults to system timezone",
            },
        },
        "required": [],
    }

    def __init__(self):
        """Initialize the datetime tool."""
        logger.debug("DateTimeTool initialized")

    async def execute(self, **kwargs) -> ToolExecutionResult:
        """Execute the datetime tool to get current date/time.

        Args:
            format: Output format ("iso", "readable", "date", "time", "timestamp")
            timezone: Optional timezone name

        Returns:
            ToolExecutionResult: Current date/time information or error
        """
        format_type = kwargs.get("format", "readable").lower()
        timezone_name = kwargs.get("timezone", "").strip()

        try:
            return await self._get_current_datetime(format_type, timezone_name)

        except Exception as e:
            logger.exception("Error getting current date/time")
            return ToolExecutionResult(
                success=False, data="", error=f"Failed to get date/time: {e!s}"
            )

    async def _get_current_datetime(
        self, format_type: str, timezone_name: str
    ) -> ToolExecutionResult:
        """Get current date/time in specified format."""
        try:
            # Get current datetime
            now = datetime.now()

            # Handle timezone if specified
            if timezone_name:
                try:
                    if zoneinfo is not None:
                        tz = zoneinfo.ZoneInfo(timezone_name)
                        now = now.replace(tzinfo=tz)
                        tz_display = timezone_name
                    else:
                        # Fallback for Python < 3.9
                        logger.warning("zoneinfo not available, using system timezone")
                        tz_display = "system"
                except Exception as e:
                    logger.warning(f"Invalid timezone '{timezone_name}': {e}")
                    tz_display = "system"
            else:
                tz_display = "system"

            # Format output based on requested format
            if format_type == "iso":
                formatted_time = now.isoformat()
                result = f"Current time (ISO 8601): {formatted_time}"
            elif format_type == "date":
                formatted_time = now.strftime("%Y-%m-%d")
                result = f"Current date: {formatted_time}"
            elif format_type == "time":
                formatted_time = now.strftime("%H:%M:%S")
                result = f"Current time: {formatted_time}"
            elif format_type == "timestamp":
                timestamp = int(now.timestamp())
                result = f"Current Unix timestamp: {timestamp}"
            else:  # readable (default)
                formatted_time = now.strftime("%A, %B %d, %Y at %I:%M:%S %p")
                result = f"Current date and time: {formatted_time}"

            # Add timezone info if specified
            if timezone_name and tz_display != "system":
                result += f" ({tz_display})"

            return ToolExecutionResult(
                success=True,
                data=result,
                metadata={
                    "timestamp": now.timestamp(),
                    "iso_format": now.isoformat(),
                    "timezone": tz_display,
                    "format_requested": format_type,
                },
            )

        except Exception as e:
            logger.exception(f"Error formatting datetime with format '{format_type}'")
            return ToolExecutionResult(
                success=False, data="", error=f"Error formatting date/time: {e!s}"
            )
