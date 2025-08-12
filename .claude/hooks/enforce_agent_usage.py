#!/usr/bin/env python3
"""
Hook to enforce agent usage for all tool calls in the project-watch-mcp project.

This hook runs before every tool use and checks if the appropriate agent
is being used for the task. If not using an agent when required, it blocks
the action and provides guidance.
"""

import json
import sys
import os
from pathlib import Path

# Tools that are allowed without agent delegation
ALLOWED_DIRECT_TOOLS = {
    "Read",
    "LS", 
    "Bash",
    "WebFetch",
    "WebSearch",
    "mcp__memory__read_graph",
    "mcp__memory__search_memories",
    "mcp__memory__find_memories_by_name"
}

# Tools that require agent delegation
REQUIRE_AGENT_TOOLS = {
    "Write",
    "Edit",
    "MultiEdit",
    "NotebookEdit",
    "TodoWrite",
    "mcp__memory__create_entities",
    "mcp__memory__create_relations",
    "mcp__memory__add_observations",
    "mcp__memory__delete_entities",
    "mcp__memory__delete_observations",
    "mcp__memory__delete_relations",
    "mcp__project-watch-local__initialize_repository",
    "mcp__project-watch-local__search_code",
    "mcp__project-watch-local__refresh_file",
    "mcp__project-watch-local__delete_file",
    "mcp__project-watch-local__analyze_complexity"
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
            # Agent is being used - allow this and mark context
            sys.exit(0)
        
        # Check if we're already in an agent context
        # This could be tracked via session state or by checking recent Task calls
        # For now, we'll check if the current working directory suggests agent context
        is_in_agent = False
        
        # If tool requires agent and we're not in agent context
        if tool_name in REQUIRE_AGENT_TOOLS and not is_in_agent:
            # Create proper denial message with JSON output
            denial_reason = f"""The tool '{tool_name}' requires delegation to an appropriate agent.

Please use one of these agents:
• @agent-project-context-expert - For project info, conventions, commands
• @agent-project-memory-navigator - For finding files, searching code  
• @agent-project-todo-orchestrator - For managing tasks and todos

Check .claude/commands/available-agents.md for the complete list."""
            
            # Output JSON decision control
            output = {
                "permissionDecision": "deny",
                "permissionDecisionReason": denial_reason
            }
            print(json.dumps(output))
            
            # Also print to stderr for visibility
            print(f"\n🚨 AGENT DELEGATION REQUIRED 🚨\n{denial_reason}", file=sys.stderr)
            
            # Exit with code 2 to block the action
            sys.exit(2)
        
        # Allow the action - no output needed for allow
        sys.exit(0)
        
    except Exception as e:
        # On error, allow the action (fail open)
        print(f"Hook error: {e}", file=sys.stderr)
        sys.exit(0)

if __name__ == "__main__":
    main()