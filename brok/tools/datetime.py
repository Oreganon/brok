"""DateTime tool for getting current date and time information."""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, ClassVar
import zoneinfo

from brok.tools.base import BaseTool, ToolExecutionResult

logger = logging.getLogger(__name__)

# Common timezone mappings for user-friendly names
TIMEZONE_MAPPINGS = {
    "new york": "America/New_York",
    "ny": "America/New_York",
    "nyc": "America/New_York",
    "eastern": "America/New_York",
    "est": "America/New_York",
    "edt": "America/New_York",
    "los angeles": "America/Los_Angeles",
    "la": "America/Los_Angeles",
    "pacific": "America/Los_Angeles",
    "pst": "America/Los_Angeles",
    "pdt": "America/Los_Angeles",
    "chicago": "America/Chicago",
    "central": "America/Chicago",
    "cst": "America/Chicago",
    "cdt": "America/Chicago",
    "denver": "America/Denver",
    "mountain": "America/Denver",
    "mst": "America/Denver",
    "mdt": "America/Denver",
    "london": "Europe/London",
    "uk": "Europe/London",
    "gmt": "GMT",
    "utc": "UTC",
    "paris": "Europe/Paris",
    "berlin": "Europe/Berlin",
    "tokyo": "Asia/Tokyo",
    "japan": "Asia/Tokyo",
    "sydney": "Australia/Sydney",
    "australia": "Australia/Sydney",
}


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
                "description": "Timezone identifier. Use IANA timezone names like 'America/New_York', 'Europe/London', 'UTC'. Common names like 'New York', 'London', 'UTC' are also supported.",
            },
        },
        "required": [],
    }

    def __init__(self) -> None:
        """Initialize the datetime tool."""
        super().__init__()
        logger.debug("DateTimeTool initialized")

    def _normalize_timezone(self, timezone_name: str) -> str:
        """Normalize timezone name to IANA format.

        Args:
            timezone_name: User-provided timezone name

        Returns:
            str: Normalized IANA timezone identifier

        Raises:
            ValueError: If timezone cannot be normalized or validated
        """
        if not timezone_name:
            return ""

        # Normalize input
        normalized = timezone_name.lower().strip()

        # Check common mappings first
        if normalized in TIMEZONE_MAPPINGS:
            return TIMEZONE_MAPPINGS[normalized]

        # If it's already in IANA format, validate it
        if zoneinfo is not None:
            try:
                zoneinfo.ZoneInfo(timezone_name)
                return timezone_name
            except Exception:
                pass

        # Provide helpful suggestions
        suggestions = []
        for user_name, iana_name in TIMEZONE_MAPPINGS.items():
            if normalized in user_name or user_name in normalized:
                suggestions.append(f"'{user_name}' -> {iana_name}")

        if suggestions:
            suggestions_text = "\n".join(suggestions[:3])  # Show top 3
            raise ValueError(
                f"Invalid timezone '{timezone_name}'. Did you mean one of these?\n{suggestions_text}\n"
                f"Or use a valid IANA timezone name like 'America/New_York', 'Europe/London', 'UTC'."
            )
        else:
            common_examples = ["America/New_York", "Europe/London", "Asia/Tokyo", "UTC"]
            raise ValueError(
                f"Invalid timezone '{timezone_name}'. "
                f"Please use a valid IANA timezone name like: {', '.join(common_examples)}"
            )

    async def execute(self, **kwargs: Any) -> ToolExecutionResult:
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
            # Validate and normalize timezone if provided
            if timezone_name:
                try:
                    timezone_name = self._normalize_timezone(timezone_name)
                except ValueError as e:
                    logger.warning(f"Timezone validation failed: {e}")
                    return ToolExecutionResult(success=False, data="", error=str(e))

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
                    tz = zoneinfo.ZoneInfo(timezone_name)
                    now = now.replace(tzinfo=tz)
                    tz_display = timezone_name
                except Exception as e:
                    # This should not happen after normalization, but just in case
                    logger.exception(
                        f"Failed to apply validated timezone '{timezone_name}'"
                    )
                    return ToolExecutionResult(
                        success=False,
                        data="",
                        error=f"Failed to apply timezone '{timezone_name}': {e}",
                    )
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
