# Claude Code Hooks

This directory contains project-specific hooks for the project-watch-mcp project, following the official Claude Code hooks specification.

## Hook Configuration

Hooks are configured in `.claude/settings.json` and follow the Claude Code hooks specification:
- https://docs.anthropic.com/en/docs/claude-code/hooks

## Active Hooks

### 1. Session Start Hook (`session_start.sh`)
- **Event**: SessionStart
- **Purpose**: Automatically initializes the MCP server when a Claude session starts
- **Behavior**: Runs `uv run project-watch-mcp --initialize` to index the repository

### 2. Agent Enforcement Hook (`enforce_agent_usage.py`)
- **Event**: PreToolUse
- **Purpose**: Enforces the use of appropriate agents for tool calls
- **Behavior**: 
  - Allows direct use of read-only tools (Read, LS, Bash, etc.)
  - Blocks write operations without proper agent delegation
  - Provides guidance on which agent to use

## Hook Structure

According to the official documentation, hooks:
1. Receive JSON input via stdin
2. Communicate via exit codes:
   - Exit 0: Allow the action to proceed
   - Exit 2: Block the action
3. Can output messages to stderr for user feedback

## Configuration Format

Hooks are configured in `.claude/settings.json`:

```json
{
  "hooks": {
    "EventName": [
      {
        "matcher": "pattern",
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/script.sh",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

## Environment Variables

- `$CLAUDE_PROJECT_DIR`: Path to the project root
- `$CLAUDE_AGENT_CONTEXT`: Custom variable to track agent context

## Testing Hooks

To test a hook manually:

```bash
# Test the enforcement hook
echo '{"tool": {"name": "Write"}}' | python3 .claude/hooks/enforce_agent_usage.py

# Test the session start hook
./session_start.sh
```

## Best Practices

1. **Fail Open**: On errors, allow the action to proceed (exit 0)
2. **Clear Messages**: Provide helpful error messages via stderr
3. **Fast Execution**: Keep timeouts short (5-30 seconds)
4. **Executable Permissions**: Ensure all hook scripts are executable
5. **Error Handling**: Wrap logic in try/catch to prevent crashes

## Adding New Hooks

1. Create the hook script in this directory
2. Make it executable: `chmod +x script.py`
3. Add configuration to `.claude/settings.json`
4. Test the hook manually before use

## Supported Hook Events

- **PreToolUse**: Before any tool is executed
- **PostToolUse**: After a tool completes
- **SessionStart**: When a Claude session begins
- **UserPromptSubmit**: When user submits a prompt
- **Stop**: When session stops
- **SubagentStop**: When a subagent stops
- **PreCompact**: Before conversation compaction
- **Notification**: For notifications

## Troubleshooting

If hooks aren't working:
1. Check file permissions (`ls -la`)
2. Verify JSON configuration syntax
3. Test hooks manually with sample input
4. Check Claude Code logs for errors
5. Ensure `$CLAUDE_PROJECT_DIR` is set correctly