# Monitoring Persistence Implementation

## Overview

This document describes the implementation of persistent monitoring for the Project Watch MCP server. The monitoring system ensures that file watching continues in the background after repository initialization.

## Problem Statement

Previously, both the CLI `--initialize` flag and the MCP tool `initialize_repository` would initialize the repository but the monitoring would stop immediately when the process exited. This meant that file changes were not being tracked continuously.

## Solution

### 1. Core Components

#### MonitoringManager (`src/project_watch_mcp/core/monitoring_manager.py`)

A new singleton-based manager that handles persistent monitoring:

- **Persistent Monitoring**: Keeps monitoring tasks running in the background
- **Instance Tracking**: Maintains a registry of active monitoring instances by project name
- **Graceful Shutdown**: Provides methods to stop individual or all monitors
- **Process Management**: Handles monitoring loop that processes file changes

Key features:
- `start_persistent_monitoring()`: Starts monitoring that continues in background
- `is_running()`: Checks if monitoring is active
- `shutdown_all()`: Stops all monitoring instances
- Signal handlers for graceful shutdown on SIGINT/SIGTERM

#### Enhanced RepositoryInitializer

The `RepositoryInitializer` now accepts a `persistent_monitoring` parameter:

```python
async def initialize(self, persistent_monitoring: bool = False) -> InitializationResult:
    # ... initialization logic ...
    
    if persistent_monitoring:
        # Use monitoring manager for persistent monitoring
        manager = MonitoringManager(...)
        monitoring_started = await manager.start_persistent_monitoring()
    else:
        # Normal monitoring (stops when process exits)
        await self._repository_monitor.start()
```

#### Updated RepositoryMonitor

Added `daemon` parameter to the `start()` method for better control:

```python
async def start(self, daemon: bool = False):
    """Start monitoring with optional daemon mode."""
```

### 2. Integration Points

#### CLI Integration (`src/project_watch_mcp/cli.py`)

The `initialize_only` function now uses persistent monitoring:

```python
async with initializer:
    result = await initializer.initialize(persistent_monitoring=True)
```

#### MCP Server Integration (`src/project_watch_mcp/server.py`)

The `initialize_repository` tool also uses persistent monitoring:

```python
async with initializer:
    result = await initializer.initialize(persistent_monitoring=True)
```

### 3. Testing

Comprehensive test coverage includes:

#### Unit Tests (`tests/test_monitoring_persistence.py`)
- Background task creation
- Monitoring persistence after initialization
- Change detection continuity
- Multiple initialization cycles
- Concurrent start handling

#### CLI Tests (`tests/test_cli_monitoring.py`)
- CLI initialization with monitoring
- Verbose output verification
- Error handling
- Singleton behavior

#### Integration Tests (`tests/test_integration_monitoring.py`)
- Full initialization flow
- Multiple project monitoring
- Restart after failure
- Batch shutdown

## Usage

### CLI Initialization

```bash
# Initialize repository with persistent monitoring
uv run project-watch-mcp --initialize --repository /path/to/repo

# Output shows monitoring status
Project: my-project
Indexed: 42/45 files
Monitoring: started
```

### MCP Tool

When using the MCP server, call the `initialize_repository` tool:

```python
# The tool automatically starts persistent monitoring
await initialize_repository()
# Returns: {"indexed": 42, "total": 45, "monitoring": true}
```

### Monitoring Status

Check if monitoring is active:

```python
from project_watch_mcp.core import MonitoringManager

# Check if project is being monitored
is_active = MonitoringManager.is_monitoring("my-project")

# Get monitoring instance
manager = MonitoringManager.get_instance("my-project")
if manager and manager.is_running():
    print("Monitoring is active")
```

## Implementation Details

### Background Task Management

The monitoring runs as an asyncio task that:
1. Continuously checks for file changes
2. Processes pending changes in batches
3. Logs activity for debugging
4. Handles graceful shutdown

### Resource Management

- Monitors are registered globally by project name
- Cleanup happens automatically on stop
- Signal handlers ensure clean shutdown
- Tasks are properly cancelled to avoid resource leaks

### Error Handling

- Failed monitoring starts are logged but don't crash initialization
- Unexpected monitor stops are detected and logged
- Path resolution handles symlinks (important for temp directories)

## Benefits

1. **Continuous Monitoring**: Files are watched continuously after initialization
2. **Multiple Projects**: Can monitor multiple projects simultaneously
3. **Graceful Shutdown**: Clean shutdown on signals or manual stop
4. **Restart Capability**: Monitoring can be restarted after failures
5. **Resource Efficient**: Uses asyncio tasks, not separate processes

## Future Enhancements

Potential improvements for future versions:

1. **True Daemon Process**: Implement as a system service or daemon
2. **Persistence Across Restarts**: Save monitoring state to disk
3. **Web Dashboard**: UI for monitoring status and control
4. **Metrics Collection**: Track file change patterns and frequency
5. **Webhook Integration**: Notify external systems of changes