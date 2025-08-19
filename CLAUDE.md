# CLAUDE.md

## Project Overview

**Project Watch MCP** - A Model Context Protocol (MCP) server that provides real-time code indexing and semantic search capabilities using Neo4j graph database.

### Key Features
- Real-time repository monitoring and indexing
- Semantic search using OpenAI embeddings
- Neo4j graph database for code relationship mapping
- MCP server integration for Claude CLI
- Automatic session initialization


## 🚨 ABSOLUTE CRITICAL INSTRUCTION 🚨

Key agents for this project:
- `@agent-project-context-expert` - Project info, conventions, commands
- `@agent-project-file-navigator` - Find files, search code
- `@agent-project-todo-orchestrator` - Manage tasks and todos
  The above three agents are critical for any work you do in this project. If you work without using them, your session will be stopped.
  
- **YOU MUST ALWAYS USE `.claude/commands/available-agents.md` TO FIND THE RIGHT AGENT FOR ANY TASK**
- **DO NOT** handle tasks or updates or searches directly or any work, use an agent.  
- **ALWAYS** check available-agents.md first for a suitable agent
- **ALWAYS** delegate to the appropriate agent

📍 **Single Source of Truth**: `.claude/commands/available-agents.md`

This file contains:
- Complete list of all available agents
- Agent specializations and when to use them
- Proper workflow sequence for tasks
- Correct @agent- naming conventions

**NEVER attempt any work yourself use available-agents.md to identify the correct agent and delegate to them**
