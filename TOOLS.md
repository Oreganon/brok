# Tool System

The brok chatbot now supports tool calling, allowing it to access external APIs and services to provide real-time information.

## Available Tools

### Weather Tool
Get current weather information for any city using the wttr.in service.

**Usage examples:**
- "What's the weather in London?"
- "Check the weather in New York"
- "Temperature in Tokyo"
- `{"tool": "weather", "params": {"city": "Paris"}}`

**Configuration:**
- No API key required - uses the free wttr.in service
- Supports any location name, city, or even IP addresses

### Calculator Tool
Perform mathematical calculations and evaluate expressions.

**Usage examples:**
- "Calculate 2 + 3 * 4"
- "What's the square root of 16?"
- "Solve 2^8"
- `{"tool": "calculator", "params": {"expression": "sin(pi/2)"}}`

**Supported operations:**
- Basic arithmetic: `+`, `-`, `*`, `/`, `**`, `%`
- Functions: `sqrt()`, `sin()`, `cos()`, `tan()`, `log()`, `exp()`, `abs()`, etc.
- Constants: `pi`, `e`, `tau`

## How It Works

1. **Tool Detection**: The bot analyzes LLM responses for tool calls in multiple formats:
   - JSON format: `{"tool": "weather", "params": {"city": "London"}}`
   - Natural language: "Let me check the weather in London"
   - Explicit format: `[TOOL:weather] city=London [/TOOL]`

2. **Tool Execution**: When a tool call is detected, the bot:
   - Validates the tool name and parameters
   - Executes the tool asynchronously
   - Returns the result to the chat

3. **Error Handling**: If tool execution fails, the bot provides a helpful error message

## Configuration

### Environment Variables

```bash
# Enable/disable tool system (default: true)
ENABLE_TOOLS=true

# Example .env file
ENABLE_TOOLS=true
```

### Programmatic Configuration

```python
from brok.config import BotConfig

config = BotConfig(
    enable_tools=True
)
```

## Adding New Tools

To create a new tool:

1. **Create the tool class**:

```python
from brok.tools.base import BaseTool, ToolExecutionResult

class MyTool(BaseTool):
    name = "my_tool"
    description = "Description of what the tool does"
    parameters = {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "Parameter description"}
        },
        "required": ["param1"]
    }
    
    async def execute(self, **kwargs) -> ToolExecutionResult:
        param1 = kwargs.get("param1")
        # Tool logic here
        return ToolExecutionResult(
            success=True,
            data=f"Result: {param1}"
        )
```

2. **Register the tool**:

```python
# In your bot setup
from brok.tools import ToolRegistry
from my_tools import MyTool

registry = ToolRegistry()
registry.register_tool(MyTool())
```

3. **Update natural language parsing** (optional):
   - Add patterns to `ToolParser._parse_natural_language()` for better UX

## Tool Response Formats

Tools return `ToolExecutionResult` objects with:
- `success`: Boolean indicating if execution succeeded
- `data`: String result to show to users
- `error`: Error message if execution failed
- `metadata`: Optional additional information

## Testing

Run the tool tests:

```bash
pytest tests/test_tools.py -v
```

## Example Interactions

**Weather Query:**
```
User: bot what's the weather in London?
Bot: Weather in London: ⛅️ +18°C
```

**Calculator Query:**
```
User: bot calculate 15% of 200
Bot: 15/100 * 200 = 30
```

**JSON Format:**
```
User: {"tool": "weather", "params": {"city": "Tokyo"}}
Bot: Weather in Tokyo: 🌧 +25°C
```

## Troubleshooting

### Tools Not Working
1. Check if tools are enabled: `ENABLE_TOOLS=true`
2. Verify bot logs for tool setup messages
3. Test with simple queries like "what's 2+2?"

### Weather Tool Issues
1. No API key required - uses the free wttr.in service
2. If weather requests fail, check internet connectivity
3. The service supports various location formats: cities, countries, coordinates

### Adding Custom Tools
1. Ensure tool inherits from `BaseTool`
2. Define required metadata: `name`, `description`, `parameters`
3. Implement async `execute()` method
4. Register tool in bot setup