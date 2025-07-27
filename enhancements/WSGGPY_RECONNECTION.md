# wsggpy Auto-Reconnection Integration

This document describes the integration of wsggpy's new auto-reconnection functionality into brok, which solves the WebSocket connection persistence bug.

## Overview

Previously, brok experienced connection drops after extended periods without a reliable reconnection mechanism. The updated wsggpy library now provides built-in auto-reconnection capabilities that we've integrated at multiple levels.

## New Features

### 1. wsggpy Auto-Reconnection

- **Automatic detection** of connection loss
- **Configurable retry attempts** and delays
- **Exponential backoff** support
- **Event-driven reconnection** handling

### 2. Enhanced Connection Monitoring

- **Real-time status tracking** via wsggpy connection info
- **Stale connection detection** (no messages for extended periods)
- **Manual intervention** when auto-reconnection fails
- **Detailed logging** of connection events

### 3. Two-Level Reconnection Strategy

1. **Primary**: wsggpy handles immediate WebSocket reconnections
2. **Secondary**: ChatBot provides oversight and manual intervention

## Configuration

### Environment Variables

```bash
# wsggpy auto-reconnection settings
WSGGPY_AUTO_RECONNECT=true          # Enable auto-reconnection (default: true)
WSGGPY_RECONNECT_ATTEMPTS=5         # Number of attempts (default: 5)
WSGGPY_RECONNECT_DELAY=2.0          # Initial delay in seconds (default: 2.0)
WSGGPY_RECONNECT_BACKOFF=true       # Use exponential backoff (default: true)

# Bot-level monitoring (still used for oversight)
CONNECTION_CHECK_INTERVAL=10        # Check interval in seconds (default: 10)
MAX_RECONNECT_ATTEMPTS=10          # Max bot-level attempts (default: 10)
```

### Code Configuration

```python
from brok.config import BotConfig
from brok.chat import ChatClient

# Load configuration with reconnection settings
config = BotConfig.from_env()

# Create chat client with auto-reconnection
chat_client = ChatClient(
    response_filters=filters,
    wsggpy_auto_reconnect=True,
    wsggpy_reconnect_attempts=5,
    wsggpy_reconnect_delay=2.0,
    wsggpy_reconnect_backoff=True
)
```

## API Reference

### ChatClient Methods

#### Connection Status

```python
# Check if connected (uses wsggpy status if available)
is_connected = chat_client.is_connected()

# Check if reconnection is in progress
is_reconnecting = chat_client.is_reconnecting()

# Get detailed connection information
conn_info = chat_client.get_connection_info()
# Returns: {'connected': bool, 'reconnecting': bool, 'connection_attempts': int, 'last_error': str}
```

#### Manual Control

```python
# Force a reconnection attempt
await chat_client.force_reconnect()
```

### Event Handlers

The ChatClient automatically handles wsggpy reconnection events:

- `_on_disconnect()` - Connection lost
- `_on_reconnecting()` - Reconnection attempt started
- `_on_reconnected()` - Reconnection successful
- `_on_reconnect_failed()` - All reconnection attempts failed

## How It Solves the Bug

### Previous Issue

- WebSocket connections would silently fail after periods of inactivity
- No automatic reconnection mechanism
- Manual monitoring required periodic connection checks
- Reconnection logic was complex and error-prone

### New Solution

1. **Immediate Detection**: wsggpy detects connection loss at the WebSocket level
2. **Automatic Recovery**: Built-in reconnection with configurable retry logic
3. **Event-Driven Updates**: Real-time notification of connection state changes
4. **Fallback Protection**: Bot-level monitoring provides additional safety net
5. **Stale Connection Detection**: Identifies connections that appear connected but aren't receiving data

## Logging Examples

With the new integration, you'll see detailed reconnection logs:

```
2024-01-15 10:30:15 - brok.chat - INFO - Configured wsggpy auto-reconnection: 5 attempts, 2.0s initial delay
2024-01-15 10:35:22 - brok.chat - WARNING - 🔌 Chat connection lost - wsggpy detected disconnection
2024-01-15 10:35:23 - brok.chat - INFO - 🔄 Chat reconnection attempt in progress...
2024-01-15 10:35:25 - brok.chat - INFO - ✅ Chat reconnection successful!
2024-01-15 11:00:30 - brok.bot - WARNING - No chat activity for 300s. Connection may be stale, forcing reconnection...
```

## Testing

Use the provided test script to verify reconnection functionality:

```bash
python test_reconnection.py
```

The test script will:

1. Establish a connection
2. Check connection status
3. Force manual reconnection
4. Monitor automatic behavior

## Migration from Previous Version

The new functionality is **backward compatible**. Existing configurations will work, but you can enable enhanced reconnection by:

1. **Update Environment Variables**: Add the new `WSGGPY_*` variables
2. **Review Connection Settings**: Adjust timeouts and retry counts as needed
3. **Monitor Logs**: Watch for the new reconnection event messages

## Troubleshooting

### Connection Still Dropping?

1. Check wsggpy version supports the new API
2. Verify `WSGGPY_AUTO_RECONNECT=true` is set
3. Increase `WSGGPY_RECONNECT_ATTEMPTS` if needed
4. Check network stability and firewall settings

### Too Many Reconnection Attempts?

1. Reduce `WSGGPY_RECONNECT_ATTEMPTS`
2. Increase `WSGGPY_RECONNECT_DELAY`
3. Enable `WSGGPY_RECONNECT_BACKOFF=true` for exponential delays

### Debug Connection Issues

1. Set `LOG_LEVEL=DEBUG` for detailed logging
2. Monitor `get_connection_info()` output
3. Check bot statistics for reconnection counts

## Performance Impact

The new reconnection system is designed to be lightweight:

- **Minimal overhead**: Event-driven rather than polling-based
- **Efficient detection**: Uses WebSocket-level signals
- **Smart backoff**: Prevents connection spam during outages
- **Resource-aware**: Configurable limits prevent runaway reconnection attempts

## Future Enhancements

Planned improvements include:

- Connection quality metrics
- Adaptive reconnection delays based on success rates
- Integration with health monitoring systems
- Websocket ping/pong health checks
