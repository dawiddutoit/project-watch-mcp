#!/usr/bin/env python3
"""
Hook to guide agent usage for tool calls in the project-watch-mcp project.

This hook runs before every tool use and provides guidance about using agents
for complex tasks, while allowing essential operations to proceed.
"""

import json
import sys
import os
from pathlib import Path

# Tools that are always allowed without agent delegation
ALLOWED_DIRECT_TOOLS = {
    "Read",
    "TodoWrite",  # Allow direct todo management
    "ExitPlanMode",
    # MCP memory tools - allow for context gathering
    "mcp__memory__read_graph",
    "mcp__memory__search_memories",
    "mcp__memory__find_memories_by_name",
    # MCP project-watch tools - allow for testing and debugging
    "mcp__project-watch-mcp__initialize_repository",
    "mcp__project-watch-mcp__search_code",
    "mcp__project-watch-mcp__get_repository_stats",
    "mcp__project-watch-mcp__get_file_info",
    "mcp__project-watch-mcp__refresh_file",
    "mcp__project-watch-mcp__delete_file",
    "mcp__project-watch-mcp__analyze_complexity",
    "mcp__project-watch-mcp__monitoring_status",
}

# Tools that suggest agent delegation for complex tasks
SUGGEST_AGENT_TOOLS = {
    "mcp__memory__create_entities",
    "mcp__memory__create_relations",
    "mcp__memory__add_observations",
    "mcp__memory__delete_entities",
    "mcp__memory__delete_observations",
    "mcp__memory__delete_relations",
}

# Agent-related tools that indicate proper delegation
AGENT_TOOLS = {"Task"}

def main():
    """Main hook logic."""
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)
        
        # Extract tool information based on PreToolUse input format
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})
        
        # Check if this is a Task tool (agent delegation)
        if tool_name == "Task":
            # Agent is being used - allow this
            sys.exit(0)
        
        # Check if tool is in the always-allowed list
        if tool_name in ALLOWED_DIRECT_TOOLS:
            # Allow direct use of these tools
            sys.exit(0)
        
        # Track if we're in a testing/debugging context
        # Look for indicators in tool_input that suggest testing or debugging
        is_testing = any(
            keyword in str(tool_input).lower() 
            for keyword in ["test", "debug", "check", "verify", "inspect"]
        )
        
        # For tools that suggest agent usage, provide guidance but don't block
        if tool_name in SUGGEST_AGENT_TOOLS and not is_testing:
            # Create suggestion message (non-blocking)
            suggestion = f"""💡 Tip: For complex {tool_name} operations, consider using agents:
• @agent-project-context-expert - For project info, conventions
• @agent-project-file-navigator - For finding files, searching code  
• @agent-project-todo-orchestrator - For managing tasks

Check .claude/commands/available-agents.md for the complete list."""
            
            # Print suggestion to stderr for visibility (non-blocking)
            print(f"\n{suggestion}", file=sys.stderr)
            
            # Allow the action to proceed
            sys.exit(0)
        
        # For any other tools, allow them to proceed
        # This ensures we don't accidentally block essential operations
        sys.exit(0)
        
    except Exception as e:
        # On error, allow the action (fail open)
        print(f"Hook error: {e}", file=sys.stderr)
        sys.exit(0)

if __name__ == "__main__":
    main()