# Brok - AI Chat Bot

## Overview

**Brok** is a Rust-based AI chat bot that connects to websocket chat services (specifically strims.gg) and provides intelligent responses using local AI inference via llama.cpp server. The bot operates as a chat participant named "brok" and can respond to direct mentions while maintaining conversation context and user information.

## Features

### Core Functionality
- **WebSocket Chat Integration**: Connects to strims.gg chat via the `wsgg` library
- **Local AI Inference**: Uses llama.cpp server API with configurable models
- **Asynchronous Processing**: Multi-threaded architecture with separate workers for inference and user management
- **Message History**: Maintains rolling context of the last 10 chat messages
- **User Information Tracking**: Stores and updates user profiles based on chat interactions

### Tool Capabilities
- **Weather Information**: Fetches current weather data using wttr.in API
- **Calculator**: Performs mathematical calculations using the `bc` command-line tool
- **Smart Tool Detection**: Automatically detects when users are asking for weather or math calculations

### Advanced Features
- **Rate Limiting**: Built-in delays to prevent spam
- **Message Retry Logic**: Automatically retries failed messages
- **Echo Detection**: Prevents responding to its own messages
- **Input Sanitization**: Validates calculator expressions for security
- **Containerized Deployment**: Docker support for easy deployment

## Architecture

### Core Components

#### Main Application (`App`)
- Manages HTTP client for API calls
- Coordinates message history and pending messages
- Handles user information storage and retrieval
- Queues inference requests and user updates

#### Inference Worker
- Processes AI inference requests asynchronously
- Integrates tool calls with LLM responses
- Manages conversation context

#### User Update Worker
- Updates user information files based on chat interactions
- Runs AI-powered analysis to extract relevant user details

### Data Flow
1. WebSocket receives chat message
2. Message added to history
3. If directed at bot (`@brok`), queued for inference
4. Inference worker processes with context and tools
5. Response sent back to chat
6. User information updated if relevant users mentioned

## Configuration

### Command Line Arguments
- `--cookie` / `-c`: Path to authentication cookie file (required)
- `--dev` / `-d`: Use development environment (chat2.strims.gg) instead of production
- `--api-host`: API endpoint host and port (default: `localhost:8080`)

### Environment Requirements
- **llama.cpp server**: Must be running and accessible (default: `localhost:8080`, configurable via `--api-host`)
- **bc Calculator**: System command required for math calculations
- **Internet Access**: For weather API calls

## Dependencies

### Core Dependencies
```toml
chrono = { version = "0.4.23", features = ["clock"] }
clap = { version = "4.1.8", features = ["derive"] }
wsgg = { git = "https://github.com/Oreganon/wsgg.git", branch = "main" }
reqwest = { version = "0.11", features = ["json"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1.0", features = ["full"] }
```

### Additional Dependencies
- `ical = "0.8.0"` (for calendar functionality, currently unused)

## Installation & Usage

### Prerequisites
1. Install Rust (latest stable version)
2. Install and run llama.cpp server with your preferred model
3. Install `bc` calculator (usually pre-installed on Unix systems)
4. Obtain authentication cookie for strims.gg

### Building
```bash
cargo build --release
```

### Code Quality & Linting
Always run the following commands before committing code:

```bash
# Format code according to Rust standards
cargo fmt

# Apply clippy suggestions and fixes
cargo clippy --fix --allow-dirty

# Run all tests
cargo test

# Final clippy check (should show no warnings)
cargo clippy
```

### Running
```bash
# Production environment with default API endpoint
./target/release/brok --cookie path/to/cookie.txt

# Development environment
./target/release/brok --cookie path/to/cookie.txt --dev

# Custom API endpoint
./target/release/brok --cookie path/to/cookie.txt --api-host localhost:8080

# Remote API endpoint
./target/release/brok --cookie path/to/cookie.txt --api-host 192.168.1.100:8080
```

### Docker Deployment
```bash
# Build image
docker build -t brok .

# Run container with default API endpoint
docker run -v /path/to/cookie.txt:/cookie.txt brok --cookie /cookie.txt

# Run container with custom API endpoint
docker run -v /path/to/cookie.txt:/cookie.txt brok --cookie /cookie.txt --api-host host.docker.internal:8080
```

## Bot Behavior

### Response Patterns
- Responds only to messages that start with `@brok`
- Uses emotes in responses (`LUL`, `PeepoWeird`, `FeelsPepoMan`)
- Provides single-sentence replies in `<reply></reply>` tags
- Maintains conversational context from recent messages

### Tool Usage
- **Weather**: Triggered by messages containing "weather"
  - Example: `@brok what's the weather in London?`
- **Calculator**: Triggered by mathematical expressions
  - Example: `@brok what is 25 * 4?`

### User Information
- Automatically tracks information about mentioned users
- Stores data in `users/` directory (one file per user)
- Updates profiles based on chat context
- Limits user information to 5 sentences maximum

## Security Considerations

### Input Validation
- Calculator expressions are sanitized to prevent code injection
- Only allows mathematical characters: `0-9+-*/()^` and spaces
- Limits expression length to 200 characters
- Rejects dangerous patterns like `..` or `//`

### API Safety
- Uses local llama.cpp server instance (no external AI API keys)
- Weather API calls are made to public wttr.in service
- File operations are restricted to `users/` directory

## Development Notes

### Model Configuration
Uses llama.cpp server API
- The model is determined by what's loaded in the llama.cpp server instance
- No model field needed in requests as llama.cpp serves one model at a time
- Supports any llama.cpp-compatible model (GGUF format)

### Adding New Tools
To add new tool capabilities:
1. Add new variant to `ToolCall` enum in `src/tools.rs`
2. Implement detection logic in `ToolManager::detect_tool_call()`
3. Add execution logic in `ToolManager::execute_tool()`

The tools are now organized in a separate module (`src/tools.rs`) with the following structure:
- `ToolCall` enum: Defines available tool types
- `ToolResult` struct: Standardized tool execution results
- `ToolManager` struct: Manages tool detection and execution
- Individual tool implementations: `calculate()`, `get_weather()`

### Debugging
- Extensive debug logging throughout the application
- Enable with `RUST_LOG=debug` environment variable
- Logs include message flow, API calls, and tool executions

## File Structure
```
brok/
├── Cargo.toml          # Project dependencies and metadata
├── Cargo.lock          # Dependency lock file
├── Dockerfile          # Container build instructions
├── src/
│   ├── main.rs         # Main application code and chat integration
│   └── tools.rs        # Tool implementations (weather, calculator)
└── users/              # User information storage (created at runtime)
    ├── username1       # User profile files
    └── username2
```

## Contributing

When contributing to this project:
1. Maintain the async/await pattern for all I/O operations
2. Add appropriate debug logging for new features
3. Follow Rust best practices for error handling
4. Test tool integrations thoroughly
5. **Always run code quality checks before committing:**
   - `cargo fmt` - Format code to Rust standards
   - `cargo clippy --fix --allow-dirty` - Apply clippy suggestions
   - `cargo test` - Ensure all tests pass
   - `cargo clippy` - Verify no warnings remain
6. Update this documentation for significant changes

### Commit Guidelines

When Claude assists with changes, commits should:
- Use descriptive commit messages following conventional commit format
- Include Claude as author when Claude makes substantial code changes:
  ```bash
  git commit --author="claude <claude@anthropic.com>" -m "feat: description"
  ```
- **Manual Review Required**: Claude should NOT auto-commit changes
- User should review all changes before committing
- Separate logical changes into focused commits
- Documentation updates (like CLAUDE.md) should typically be in separate commits

## License

This project uses dependencies with various licenses. Check individual dependency licenses for compliance requirements. 

## Troubleshooting

### Multiple Response Bug

**Problem**: The LLM generates multiple example responses instead of a single answer to the user's question.

**Symptoms**:
- Very long AI responses (>500 characters) in debug logs
- Response contains multiple "User question:" and "<reply>" patterns
- Bot may respond with seemingly random content

**Root Causes & Fixes**:

1. **Prompt Engineering Issue (Fixed)**
   - **Cause**: Complex prompt with multiple examples confused small models
   - **Fix**: Simplified prompt to be more direct and concise
   - **Before**: Long prompt with multiple examples and complex instructions
   - **After**: Simple, clear instructions with minimal examples

2. **Response Validation (Added)**
   - **Protection**: Added response length validation (>500 chars triggers special handling)
   - **Recovery**: Extracts first valid reply before "User question:" patterns appear
   - **Fallback**: Returns "FeelsPepoMan" if parsing fails completely

3. **API Endpoint Debugging (Enhanced)**
   - **Issue**: Logs may show `/completion` endpoint instead of expected `/api/generate`
   - **Debug**: Added detailed request/response logging
   - **Check**: Verify your API server expects Ollama-format requests

**If issues persist**:
1. Check that your API server at `localhost:11434` is actually Ollama
2. Try a larger model (4B+ parameters) if available
3. Verify the endpoint accepts Ollama-style requests with `model`, `prompt`, `stream` fields
4. Check for any proxy/redirect between the bot and API server

### API Server Compatibility

The bot expects an Ollama-compatible API at `/api/generate` that accepts:
```json
{
  "model": "model_name",
  "prompt": "text",
  "stream": false
}
```

If your setup uses a different API format (like OpenAI-compatible), you may need to:
- Use an API proxy/adapter
- Modify the request structure in `OllamaRequest`
- Adjust the endpoint path 