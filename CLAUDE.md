# CLAUDE.md

Project Watch MCP - Neo4j-based code indexing and search MCP server.

## Quick Facts
- **Version**: 0.1.0  
- **Python**: 3.11+  
- **Package Manager**: `uv` (required)
- **Database**: Neo4j 5.11+ (local instance via Neo4j Desktop, NOT Docker)

⚠️ **Known Issue**: Semantic search currently uses mock embeddings - see todo.md for details

## Environment Setup

Neo4j connection via shell environment variables:
- `NEO4J_URI`: neo4j://127.0.0.1:7687
- `NEO4J_USER`: neo4j  
- `NEO4J_PASSWORD`: (set in shell)
- `NEO4J_DB`: memory

## Project Structure
```
src/project_watch_mcp/     # Main package
.claude/agents/            # Agent definitions  
tests/                     # Test files
```

## Critical Agent Usage Requirements

**MANDATORY: Always use these specialized agents for their domains:**

### 1. project-context-expert
- **ALWAYS USE FOR**: Project questions, configuration, setup, patterns, conventions
- **Definition**: `.claude/agents/project-context-expert.md`
- **Example**: "What testing framework does this project use?", "How is the project structured?"

### 2. project-memory-navigator  
- **ALWAYS USE FOR**: File discovery, code search, navigation, locating elements
- **Definition**: `.claude/agents/project-memory-navigator.md`
- **Example**: "Find the server.py file", "Where is the RepositoryMonitor class?"

### 3. project-todo-orchestrator
- **ALWAYS USE FOR**: Task management, todo creation/updates, work coordination
- **Definition**: `.claude/agents/project-todo-orchestrator.md`
- **Example**: "Create todos for this feature", "Update task status", "Organize work items"

## Development Commands

```bash
# Install/update dependencies
uv sync

# Run tests
uv run pytest

# Format code
uv run black src tests

# Lint code  
uv run ruff check src tests

# Start server
uv run project-watch-mcp --repository .
```

## Usage Examples

```python
# Project context and conventions
Task(
    subagent_type="project-context-expert",
    prompt="What testing framework should I use?"
)

# File search and navigation
Task(
    subagent_type="project-memory-navigator",
    prompt="find the server.py file"
)

# Task management
Task(
    subagent_type="project-todo-orchestrator",
    prompt="create todos for implementing new search feature"
)
```

## Important Instructions

- **DO NOT** attempt to handle file search, project context, or task management directly
- **ALWAYS** delegate to the appropriate agent based on the domain
- These agents have specialized knowledge and tools that the main context lacks
- Using agents reduces context usage and improves accuracy
