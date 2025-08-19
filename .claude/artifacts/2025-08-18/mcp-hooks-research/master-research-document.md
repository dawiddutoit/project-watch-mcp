# MCP Tools and SessionStart Hooks Research Report
**Date:** 2025-08-18  
**Researcher:** Strategic Research Analyst  
**Topic:** Can SessionStart Hooks Call MCP Tools in Claude Code?

## Executive Summary

### Critical Finding
**SessionStart hooks CANNOT directly call MCP tools.** Based on comprehensive analysis of the official Anthropic documentation, hooks operate as shell command executors that can modify the environment and add context, but they lack a direct mechanism to trigger MCP tool execution within Claude's runtime.

### Confidence Level: HIGH (95%)
This conclusion is based on:
- Official Anthropic documentation analysis
- Absence of any documented examples showing direct MCP tool invocation from hooks
- Clear architectural separation between hook execution and Claude's tool-calling mechanism

## Detailed Analysis

### 1. Hook Architecture and Capabilities

#### What Hooks CAN Do:
- Execute arbitrary shell commands with 60-second timeout
- Access environment variables including `$CLAUDE_PROJECT_DIR`
- Write to stdout/stderr for communication
- Modify files in the project directory
- Add context to Claude's session (SessionStart specifically)
- Validate and potentially block tool usage (PreToolUse hooks)

#### What Hooks CANNOT Do:
- Directly invoke MCP tools within Claude's execution context
- Inject tool calls into Claude's command queue
- Programmatically trigger Claude to execute specific MCP tools
- Access Claude's internal API for tool execution

### 2. SessionStart Hook Specifics

From the documentation:
> "SessionStart hooks run when starting a new session or resuming an existing one"

Key characteristics:
- **Execution Context:** Runs as a shell command in the project directory
- **Input:** Receives JSON via stdin with session metadata
- **Output:** stdout is added to context (not shown to user directly)
- **Purpose:** "Loading in development context like existing issues or recent changes"

### 3. MCP Tool Integration Reality

MCP tools follow the naming pattern: `mcp__<server>__<tool>`

While hooks can:
- **Intercept** MCP tool calls (PreToolUse/PostToolUse)
- **Match** MCP tool patterns for validation
- **Log** MCP tool usage

Hooks cannot:
- **Initiate** MCP tool calls directly
- **Queue** MCP tools for execution
- **Trigger** Claude to call specific MCP tools

### 4. The Architectural Gap

The documentation reveals a fundamental architectural separation:

```
┌─────────────┐       ┌──────────────┐       ┌─────────────┐
│    Hooks    │  ───> │ Shell Cmds   │  ───> │  File I/O   │
└─────────────┘       └──────────────┘       └─────────────┘
                            ↓
                     [Context/Validation]
                            ↓
┌─────────────┐       ┌──────────────┐       ┌─────────────┐
│   Claude    │  ───> │  MCP Tools   │  ───> │ MCP Servers │
└─────────────┘       └──────────────┘       └─────────────┘
```

Hooks operate at the shell/filesystem level, while MCP tools operate within Claude's runtime.

## Alternative Approaches Identified

### 1. Indirect Initialization via Context Injection
**Strategy:** Use SessionStart to add context that prompts Claude to initialize MCP tools

```bash
# In SessionStart hook
echo "IMPORTANT: Please initialize project-watch-mcp by calling the initialization tool."
```

**Confidence:** Medium (60%) - Relies on Claude following instructions

### 2. File-Based Signaling
**Strategy:** SessionStart creates a flag file that Claude checks

```bash
# In SessionStart hook
echo "true" > .claude/needs-initialization
```

Then rely on Claude checking this file and acting accordingly.

**Confidence:** Low (40%) - Requires consistent Claude behavior

### 3. Direct MCP Server Initialization
**Strategy:** Start the MCP server process directly from the hook

```bash
# In SessionStart hook
if ! pgrep -f "project-watch-mcp"; then
    nohup project-watch-mcp serve > /dev/null 2>&1 &
fi
```

**Confidence:** High (85%) - Direct process management

### 4. Hybrid Approach (RECOMMENDED)
**Strategy:** Combine direct server startup with context injection

```bash
#!/bin/bash
# SessionStart hook

# 1. Ensure MCP server is running
if ! pgrep -f "project-watch-mcp"; then
    project-watch-mcp serve &
    sleep 2
fi

# 2. Add initialization reminder to context
cat << EOF
PROJECT-WATCH-MCP INITIALIZATION REQUIRED:
Please execute: mcp__project-watch__initialize_project
Repository: /Users/dawiddutoit/projects/play/project-watch-mcp
EOF
```

**Confidence:** High (90%) - Combines multiple strategies

## Risk Assessment

### Critical Risks Identified:
1. **Assumption Risk:** Assuming hooks can call MCP tools directly would lead to implementation failure
2. **Dependency Risk:** Relying solely on Claude following context instructions is unreliable
3. **Process Management Risk:** Direct server startup might conflict with existing instances

### Mitigation Strategies:
1. Use process checking before starting servers
2. Implement idempotent initialization
3. Add error handling and logging
4. Provide clear user feedback if initialization fails

## Evidence and Citations

### Direct Quotes from Documentation:

1. On hook capabilities:
> "Hooks are configurable scripts that can execute at specific events during Claude Code's workflow"

2. On SessionStart purpose:
> "Useful for loading in development context like existing issues or recent changes"

3. On hook execution:
> "Hooks provide deterministic control over Claude Code's behavior, ensuring certain actions always happen rather than relying on the LLM to choose to run them"

### Notable Absence:
No documentation exists showing:
- Hooks calling MCP tools
- Hooks triggering Claude tool execution
- Direct hook-to-MCP communication

## Recommendations

### Primary Recommendation:
**DO NOT attempt to call MCP tools directly from SessionStart hooks.**

### Suggested Implementation:
1. **Immediate:** Use the hybrid approach combining server startup with context injection
2. **Short-term:** Implement robust process management in the hook
3. **Long-term:** Consider requesting this feature from Anthropic

### Implementation Code:

```bash
#!/bin/bash
# .claude/hooks/session-start.sh

set -e

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
LOG_FILE="$PROJECT_DIR/.claude/logs/session-start.log"
MCP_SERVER="project-watch-mcp"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Log session start
echo "[$(date)] SessionStart hook triggered" >> "$LOG_FILE"

# Check if MCP server is configured
if ! claude mcp list 2>/dev/null | grep -q "$MCP_SERVER"; then
    echo "[$(date)] WARNING: $MCP_SERVER not configured" >> "$LOG_FILE"
    echo "WARNING: project-watch-mcp MCP server not configured. Please run: claude mcp add project-watch"
    exit 0
fi

# Output context for Claude
cat << EOF
=== Project Watch MCP Initialization Required ===

This project uses project-watch-mcp for code indexing and search.

MANUAL INITIALIZATION REQUIRED:
Please execute the following MCP tool to initialize the project:
- Tool: mcp__project-watch__initialize_project
- Repository: $PROJECT_DIR

This will:
1. Connect to Neo4j database
2. Index the repository
3. Enable semantic search capabilities

Please confirm initialization by running the tool now.
EOF

echo "[$(date)] Context injection completed" >> "$LOG_FILE"
exit 0
```

## Confidence Levels Summary

| Finding | Confidence |
|---------|------------|
| Hooks cannot directly call MCP tools | 95% |
| Context injection will prompt Claude | 60% |
| Direct server startup will work | 85% |
| Hybrid approach effectiveness | 90% |
| File-based signaling reliability | 40% |

## Next Steps

1. **Immediate Action:** Implement the hybrid approach hook
2. **Testing Required:** Verify hook execution in different scenarios
3. **Documentation:** Update project documentation with findings
4. **Feature Request:** Consider submitting feature request to Anthropic
5. **Monitoring:** Track initialization success rates

## Conclusion

SessionStart hooks **cannot** directly call MCP tools. The architecture intentionally separates hook execution (shell level) from MCP tool invocation (Claude runtime level). The recommended solution is a hybrid approach that manages the MCP server process directly while injecting context to prompt Claude for manual initialization.

This limitation appears to be by design, maintaining security boundaries between user-defined shell commands and Claude's tool execution environment.

---
*Research completed: 2025-08-18*  
*Documentation sources: Anthropic Official Documentation*  
*Confidence in conclusions: HIGH*