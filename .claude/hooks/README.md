# Claude SessionStart Hook for Project Watch MCP

This directory contains hooks that are executed when Claude starts a session in this project.

## Enhanced SessionStart Hook

The `session-start/session-start.py` hook **directly initializes** the Project Watch MCP repository monitoring when a Claude session begins, without requiring Claude to interpret and execute commands.

### What it does:

1. **Direct Execution**: Imports and runs the server initialization code directly
2. **Connects to Neo4j**: Establishes database connection with 5-second timeout
3. **Scans Repository**: Identifies all matching files based on patterns
4. **Indexes Files**: Creates semantic embeddings for intelligent search
5. **Returns Status**: Reports success/failure to Claude with statistics

### Manual Initialization

If the automatic hook doesn't trigger or you disabled it, you can manually initialize the repository by:

1. **Using the MCP tool directly**:
   ```
   Call: mcp__project-watch-local__initialize_repository
   ```

2. **Running the hook manually**:
   ```bash
   echo '{"session_id": "manual", "cwd": "$(pwd)"}' | python .claude/hooks/session-start/session-start.py
   ```

### Disabling Auto-Initialization

To prevent automatic initialization on session start:

```bash
touch .claude/.skip_auto_init
```

Remove the file to re-enable:
```bash
rm .claude/.skip_auto_init
```

### Configuration

The hook configuration is defined in `config.json` and includes:
- Hook enablement status
- Timeout settings (30 seconds default)
- Error handling strategy (continue on error)
- Logging configuration

### Environment Variables Required

The following environment variables should be set for Neo4j connection:
- `NEO4J_URI`: Connection URI (default: neo4j://127.0.0.1:7687)
- `NEO4J_USER`: Username (default: neo4j)
- `NEO4J_PASSWORD`: Password (required, no default)
- `NEO4J_DB`: Database name (default: memory)

### Hook Execution Flow

1. Claude session starts in the project directory
2. Claude checks for `.claude/hooks/config.json`
3. If SessionStart hook is enabled, executes `session-start/session-start.py`
4. Hook directly imports server components and initializes repository
5. Repository is scanned and all files are indexed with embeddings
6. Claude receives success confirmation with statistics
7. Repository monitoring and semantic search are immediately available

### Performance

- **Direct initialization**: ~2-3 seconds for typical repository
- **67 files indexed**: ~2 seconds (including embeddings generation)
- **Fallback to manual**: Only if direct initialization fails

### Troubleshooting

If the hook fails:
1. Check that Neo4j environment variables are set
2. Ensure Neo4j Desktop is running  
3. Verify the MCP server is properly configured
4. Check `.claude/hooks/logs/` for detailed error messages
5. Look for `.claude/.last_auto_init` for last initialization result

### Files in this directory

- `session-start/session-start.py`: Enhanced SessionStart hook with direct initialization
- `utils/hook_output.py`: Utility module for formatting hook outputs
- `config.json`: Hook configuration file
- `README.md`: This documentation file
- `logs/`: Directory for detailed session logs (created at runtime)
- `.last_auto_init`: Marker file with last initialization result (created at runtime)