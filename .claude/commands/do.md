# Command: do

## Purpose
Initialize session with critical project agents and context

## Instructions

When this command is invoked at the start of a session, you MUST:

1. **Read Project Context** - Read and internalize CLAUDE.md for project-specific instructions
2. **Load Critical Agents** - Use the following agents for ALL work:
   - `@agent-project-context-expert` - For project info, conventions, and commands
   - `@agent-project-file-navigator` - For finding files and searching code  
   - `@agent-project-todo-orchestrator` - For managing tasks and todos

3. **Workflow Enforcement**:
   - NEVER handle tasks directly - always delegate to appropriate agents
   - ALWAYS check `.claude/commands/available-agents.md` for suitable agents when assigning work
   - ALWAYS use agents for searches, updates, and any project work
   - Session will be terminated if work is done without using agents

## Usage
```
/do
```

## Critical Reminders
- This is NOT optional - these agents are REQUIRED for all project work
- Direct work without agents violates project protocol
- Always delegate, never implement directly