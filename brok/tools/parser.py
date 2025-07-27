"""Parser for detecting and parsing tool calls in LLM responses."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Represents a parsed tool call from LLM response."""

    tool_name: str
    parameters: dict[str, Any]
    raw_text: str

    def __str__(self) -> str:
        """String representation of the tool call."""
        return f"ToolCall(tool='{self.tool_name}', params={self.parameters})"


class ToolParser:
    """Parser for detecting and extracting tool calls from LLM responses.

    Supports multiple formats:
    1. JSON format: {"tool": "weather", "params": {"city": "London"}}
    2. Natural language: "Let me check the weather in London"
    3. Explicit format: [TOOL:weather] city=London [/TOOL]

    Example:
        >>> parser = ToolParser(available_tools=["weather", "calculator"])
        >>> call = parser.parse_response("Let me check the weather in London")
        >>> if call:
        ...     print(f"Tool: {call.tool_name}, Params: {call.parameters}")
    """

    def __init__(self, available_tools: list[str] | None = None):
        """Initialize the parser with available tools.

        Args:
            available_tools: List of available tool names for validation
        """
        self.available_tools = set(available_tools or [])
        logger.debug(f"Initialized tool parser with tools: {self.available_tools}")

    def parse_response(self, response: str) -> ToolCall | None:
        """Parse an LLM response for tool calls.

        Args:
            response: The LLM response text to parse

        Returns:
            ToolCall | None: Parsed tool call if found, None otherwise
        """
        response = response.strip()

        # Try JSON format first
        json_call = self._parse_json_format(response)
        if json_call:
            # Validate tool availability if we have a tool list
            if self.available_tools and json_call.tool_name not in self.available_tools:
                logger.warning(
                    f"JSON format tool call uses unavailable tool '{json_call.tool_name}'. "
                    f"Available tools: {sorted(self.available_tools)}"
                )
                # Still return the call - let the registry handle the error
            return json_call

        # Try explicit tool format
        explicit_call = self._parse_explicit_format(response)
        if explicit_call:
            # Validate tool availability
            if (
                self.available_tools
                and explicit_call.tool_name not in self.available_tools
            ):
                logger.warning(
                    f"Explicit format tool call uses unavailable tool '{explicit_call.tool_name}'. "
                    f"Available tools: {sorted(self.available_tools)}"
                )
            return explicit_call

        # Try natural language detection
        nl_call = self._parse_natural_language(response)
        if nl_call:
            # Natural language parsing already checks available tools
            return nl_call

        # Check if response might be attempting a tool call
        self._check_for_failed_tool_patterns(response)

        return None

    def _parse_json_format(self, response: str) -> ToolCall | None:
        """Parse JSON format tool calls.

        Expected format: {"tool": "name", "params": {"key": "value"}}
        """
        try:
            # First try to parse the entire response as JSON
            if response.strip().startswith("{") and response.strip().endswith("}"):
                try:
                    data = json.loads(response.strip())
                    tool_name = data.get("tool", "").lower()
                    params = data.get("params", data.get("parameters", {}))

                    if tool_name and (
                        not self.available_tools or tool_name in self.available_tools
                    ):
                        logger.debug(
                            f"Parsed JSON tool call (full response): {tool_name}"
                        )
                        return ToolCall(
                            tool_name=tool_name,
                            parameters=params,
                            raw_text=response.strip(),
                        )
                except json.JSONDecodeError:
                    pass

            # Look for JSON objects within the response using improved pattern
            # Match balanced braces to handle nested objects
            json_pattern = (
                r'\{[^{}]*"tool"[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*"tool"[^{}]*\}'
            )
            matches = re.findall(json_pattern, response, re.IGNORECASE)

            for match in matches:
                try:
                    data = json.loads(match)
                    tool_name = data.get("tool", "").lower()
                    params = data.get("params", data.get("parameters", {}))

                    if tool_name and (
                        not self.available_tools or tool_name in self.available_tools
                    ):
                        logger.debug(f"Parsed JSON tool call (embedded): {tool_name}")
                        return ToolCall(
                            tool_name=tool_name, parameters=params, raw_text=match
                        )
                except json.JSONDecodeError:
                    logger.debug(f"Failed to parse JSON match: {match}")
                    continue

            # Fallback: try to find any JSON-like structure with "tool" key
            tool_pattern = r'\{[^{}]*"tool"\s*:\s*"([^"]+)"[^{}]*\}'
            tool_matches = re.findall(tool_pattern, response, re.IGNORECASE)

            if tool_matches:
                # Try to extract the full JSON object around the tool match
                for tool_name in tool_matches:
                    if tool_name.lower() in self.available_tools:
                        # Find the complete JSON object containing this tool
                        start_idx = response.find('{"tool"')
                        if start_idx == -1:
                            start_idx = response.find("{'tool'")

                        if start_idx != -1:
                            # Find the matching closing brace
                            brace_count = 0
                            end_idx = start_idx

                            for i in range(start_idx, len(response)):
                                if response[i] == "{":
                                    brace_count += 1
                                elif response[i] == "}":
                                    brace_count -= 1
                                    if brace_count == 0:
                                        end_idx = i + 1
                                        break

                            potential_json = response[start_idx:end_idx]
                            try:
                                # Normalize single quotes to double quotes for JSON
                                normalized_json = potential_json.replace("'", '"')
                                data = json.loads(normalized_json)
                                json_tool_name = data.get("tool", "").lower()
                                params = data.get("params", data.get("parameters", {}))

                                if (
                                    json_tool_name
                                    and json_tool_name in self.available_tools
                                ):
                                    logger.debug(
                                        f"Parsed JSON tool call (fallback): {json_tool_name}"
                                    )
                                    return ToolCall(
                                        tool_name=json_tool_name,
                                        parameters=params,
                                        raw_text=potential_json,
                                    )
                            except json.JSONDecodeError:
                                logger.debug(
                                    f"Fallback JSON parsing failed for: {potential_json}"
                                )
                                continue

        except Exception as e:
            logger.warning(f"Error parsing JSON format: {e}")

        return None

    def _parse_explicit_format(self, response: str) -> ToolCall | None:
        """Parse explicit tool format.

        Expected format: [TOOL:name] param1=value1 param2=value2 [/TOOL]
        """
        try:
            pattern = r"\[TOOL:([^\]]+)\]([^\[]*?)\[/TOOL\]"
            matches = re.findall(pattern, response, re.IGNORECASE)

            for tool_name_raw, params_str in matches:
                explicit_tool_name = tool_name_raw.strip().lower()

                if (
                    not self.available_tools
                    or explicit_tool_name in self.available_tools
                ):
                    # Parse parameters
                    params = self._parse_parameter_string(params_str)

                    logger.debug(f"Parsed explicit tool call: {explicit_tool_name}")
                    return ToolCall(
                        tool_name=explicit_tool_name,
                        parameters=params,
                        raw_text=f"[TOOL:{explicit_tool_name}]{params_str}[/TOOL]",
                    )

        except Exception as e:
            logger.debug(f"Error parsing explicit format: {e}")

        return None

    def _parse_natural_language(self, response: str) -> ToolCall | None:
        """Parse natural language tool requests.

        Detects tool usage based on keywords and context.
        """
        response_lower = response.lower()

        # Weather tool detection
        if "weather" in self.available_tools:
            weather_patterns = [
                r"weather\s+in\s+([a-zA-Z\s,]+)",
                r"what['']?s\s+the\s+weather\s+in\s+([a-zA-Z\s,]+)",
                r"check\s+the\s+weather\s+(?:in\s+|for\s+)?([a-zA-Z\s,]+)",
                r"temperature\s+in\s+([a-zA-Z\s,]+)",
            ]

            for pattern in weather_patterns:
                match = re.search(pattern, response_lower)
                if match:
                    city = match.group(1).strip().title()
                    logger.debug(f"Parsed natural language weather request for: {city}")
                    return ToolCall(
                        tool_name="weather",
                        parameters={"city": city},
                        raw_text=response,
                    )

        # Calculator tool detection
        if "calculator" in self.available_tools:
            calc_patterns = [
                r"calculate\s+(.+)",
                r"what['']?s\s+(.+\s*[+\-*/]\s*.+)",
                r"compute\s+(.+)",
                r"solve\s+(.+)",
            ]

            for pattern in calc_patterns:
                match = re.search(pattern, response_lower)
                if match:
                    expression = match.group(1).strip()
                    logger.debug(f"Parsed natural language calculation: {expression}")
                    return ToolCall(
                        tool_name="calculator",
                        parameters={"expression": expression},
                        raw_text=response,
                    )

        # DateTime tool detection
        if "datetime" in self.available_tools:
            datetime_patterns = [
                r"what\s+time\s+is\s+it(?:\s+in\s+([a-zA-Z][a-zA-Z\s,/]{1,30}))?",
                r"current\s+time(?:\s+in\s+([a-zA-Z][a-zA-Z\s,/]{1,30}))?",
                r"what['']?s\s+(?:the\s+)?current\s+date",  # Handle contractions
                r"what\s+(?:is\s+)?(?:the\s+)?current\s+date",
                r"what['']?s\s+(?:the\s+)?date",  # Handle contractions
                r"what\s+(?:is\s+)?(?:the\s+)?date",
                r"get\s+(?:me\s+)?(?:the\s+)?(?:current\s+)?(?:date\s+and\s+)?time",
                r"show\s+(?:me\s+)?(?:the\s+)?(?:current\s+)?(?:date\s+and\s+)?time",
                r"tell\s+me\s+(?:the\s+)?(?:current\s+)?(?:date\s+and\s+)?time",
                r"(?:current\s+)?datetime",
                # More precise time zone pattern - only match common timezone formats
                r"(?:what\s+)?time\s+in\s+((?:new\s+york|london|tokyo|paris|berlin|sydney|utc|est|pst|cst|mst|gmt|[a-z]{3,4})\b)",
            ]

            for pattern in datetime_patterns:
                match = re.search(pattern, response_lower)
                if match:
                    # Check if timezone was captured
                    if match.groups() and match.group(1):
                        timezone = match.group(1).strip()

                        # Additional validation: timezone should be reasonable length and format
                        if self._is_valid_timezone_request(timezone):
                            logger.debug(
                                f"Parsed natural language datetime request with timezone: {timezone}"
                            )
                            return ToolCall(
                                tool_name="datetime",
                                parameters={"timezone": timezone},
                                raw_text=response,
                            )
                        else:
                            logger.debug(
                                f"Rejected invalid timezone in datetime request: '{timezone}'"
                            )
                    else:
                        logger.debug("Parsed natural language datetime request")
                        return ToolCall(
                            tool_name="datetime",
                            parameters={},
                            raw_text=response,
                        )

        return None

    def _parse_parameter_string(self, params_str: str) -> dict[str, Any]:
        """Parse parameter string into dictionary.

        Expected format: param1=value1 param2="value with spaces" param3=123
        """
        params: dict[str, Any] = {}

        # Pattern to match key=value pairs, handling quoted values
        pattern = r'(\w+)=(?:"([^"]*)"|([^\s]+))'
        matches = re.findall(pattern, params_str)

        for key, quoted_value, unquoted_value in matches:
            value = quoted_value if quoted_value else unquoted_value

            # Try to convert to appropriate type
            if value.isdigit():
                params[key] = int(value)
            elif value.replace(".", "", 1).isdigit():
                params[key] = float(value)
            elif value.lower() in ("true", "false"):
                params[key] = value.lower() == "true"
            else:
                params[key] = value

        return params

    def update_available_tools(self, tools: list[str]) -> None:
        """Update the list of available tools.

        Args:
            tools: New list of available tool names
        """
        self.available_tools = set(tools)
        logger.debug(f"Updated available tools: {self.available_tools}")

    def is_tool_call(self, response: str) -> bool:
        """Check if a response contains a tool call.

        Args:
            response: The response text to check

        Returns:
            bool: True if the response contains a tool call
        """
        return self.parse_response(response) is not None

    def _check_for_failed_tool_patterns(self, response: str) -> None:
        """Check for patterns that might indicate failed tool call attempts."""
        response_lower = response.lower()

        # Common tool-like patterns that might indicate confusion
        tool_indicators = [
            "tool:",
            '"tool"',
            "function:",
            "execute:",
            "call:",
            "invoke:",
            "use tool",
            "tool call",
        ]

        # Check for JSON-like structures without proper tool syntax
        json_like_patterns = [
            r'\{[^}]*"[^"]*"[^}]*\}',  # Any JSON-like structure
            r"\{[^}]*tool[^}]*\}",  # Contains 'tool' in braces
            r"\{[^}]*function[^}]*\}",  # Contains 'function' in braces
        ]

        has_tool_indicator = any(
            indicator in response_lower for indicator in tool_indicators
        )
        has_json_like = any(
            re.search(pattern, response_lower) for pattern in json_like_patterns
        )

        if has_tool_indicator or has_json_like:
            logger.info(
                f"Response contains tool-like patterns but no valid tool call was parsed. "
                f"Response snippet: '{response[:100]}...'. "
                f"Available tools: {sorted(self.available_tools) if self.available_tools else 'none'}"
            )

    def _is_valid_timezone_request(self, timezone: str) -> bool:
        """Validate that a timezone string looks like a real timezone request.

        Args:
            timezone: The timezone string to validate

        Returns:
            bool: True if it looks like a valid timezone request
        """
        timezone_lower = timezone.lower()

        # Too long to be a reasonable timezone
        if len(timezone) > 30:
            return False

        # Common timezone indicators
        valid_timezone_words = {
            "new",
            "york",
            "los",
            "angeles",
            "london",
            "tokyo",
            "paris",
            "berlin",
            "sydney",
            "chicago",
            "denver",
            "utc",
            "gmt",
            "est",
            "pst",
            "cst",
            "mst",
            "edt",
            "pdt",
            "cdt",
            "mdt",
            "america",
            "europe",
            "asia",
            "australia",
            "pacific",
            "mountain",
            "central",
            "eastern",
        }

        # Split timezone into words
        words = re.findall(r"[a-z]+", timezone_lower)

        # Must have at least one word that's a timezone indicator
        return len(words) <= 5 and any(word in valid_timezone_words for word in words)
