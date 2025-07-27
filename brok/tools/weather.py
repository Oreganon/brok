"""Weather tool for getting current weather information."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import aiohttp

from brok.tools.base import BaseTool, ToolExecutionResult

logger = logging.getLogger(__name__)


class WeatherTool(BaseTool):
    """Tool for getting current weather information.

    Uses wttr.in service to fetch weather data for specified cities.
    No API key required.

    Example:
        >>> tool = WeatherTool()
        >>> result = await tool.execute(city="London")
        >>> print(result.data)  # "Weather in London: Clear, 18°C, Light breeze"
    """

    name: ClassVar[str] = "weather"
    description: ClassVar[str] = "Get current weather information for a city"
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name (e.g., 'London', 'New York', 'Tokyo')",
            }
        },
        "required": ["city"],
    }

    def __init__(self) -> None:
        """Initialize the weather tool.

        Uses wttr.in service which requires no API key.
        Weather data is cached for 10 minutes to reduce API calls.
        """
        # Cache weather results for 10 minutes (600 seconds)
        super().__init__(cache_ttl_seconds=600)
        self.base_url = "https://wttr.in"
        logger.debug(
            "WeatherTool initialized using wttr.in service with 10-minute caching"
        )

    async def execute(self, **kwargs: Any) -> ToolExecutionResult:
        """Execute the weather tool to get current weather.

        Args:
            city: Name of the city to get weather for

        Returns:
            ToolExecutionResult: Weather information or error
        """
        city = kwargs.get("city", "").strip()

        if not city:
            return ToolExecutionResult(
                success=False, data="", error="City name is required"
            )

        try:
            return await self._fetch_weather(city)

        except Exception as e:
            logger.exception(f"Error fetching weather for {city}")
            return ToolExecutionResult(
                success=False, data="", error=f"Failed to fetch weather data: {e!s}"
            )

    async def _fetch_weather(self, city: str) -> ToolExecutionResult:
        """Fetch weather data from wttr.in service."""
        # Use format=3 for a concise one-line format
        url = f"{self.base_url}/{city}?format=3"

        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response,
            ):
                if response.status == 200:
                    weather_text = await response.text()
                    weather_text = weather_text.strip()

                    if weather_text and "Unknown location" not in weather_text:
                        # Format the response nicely
                        formatted_weather = f"Weather in {city.title()}: {weather_text}"

                        return ToolExecutionResult(
                            success=True,
                            data=formatted_weather,
                            metadata={
                                "source": "wttr.in",
                                "city": city,
                                "raw_response": weather_text,
                            },
                        )
                    else:
                        return ToolExecutionResult(
                            success=False,
                            data="",
                            error=f"City '{city}' not found or invalid location",
                        )
                else:
                    logger.warning(f"wttr.in API error {response.status}")
                    return ToolExecutionResult(
                        success=False,
                        data="",
                        error=f"Weather service returned error {response.status}",
                    )

        except aiohttp.ClientError:
            logger.exception(f"HTTP error fetching weather for {city}")
            return ToolExecutionResult(
                success=False, data="", error="Unable to connect to weather service"
            )
        except Exception as e:
            logger.exception(f"Unexpected error fetching weather for {city}")
            return ToolExecutionResult(
                success=False, data="", error=f"Unexpected error: {e!s}"
            )
