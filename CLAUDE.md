# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Project Watch MCP is a repository monitoring MCP server that creates a Neo4j-based RAG (Retrieval-Augmented Generation) system for codebases. It provides real-time file monitoring, semantic code search, and intelligent indexing capabilities through the Model Context Protocol (MCP).

**Version**: 0.1.0  
**Status**: Active development  
**Python**: 3.11+  
**Main Dependencies**: Neo4j 5.11+, FastMCP, watchfiles

## Development Setup

### Neo4j Instance
**IMPORTANT**: This project uses a **locally running Neo4j instance** (NOT Docker):
- Neo4j runs via **Neo4j Desktop** application on macOS
- Connection details are provided via shell environment variables:
  - `NEO4J_URI`: neo4j://127.0.0.1:7687
  - `NEO4J_USER`: neo4j  
  - `NEO4J_PASSWORD`: (set in shell environment)
  - `NEO4J_DB`: memory (note: uses `NEO4J_DB` not `NEO4J_DATABASE`)
- The `.env` file references these shell variables with fallback defaults
- **DO NOT** assume Docker is being used for Neo4j

### Package Management
This project uses `uv` as the package manager (preferred over pip).

```bash
# Install dependencies
uv sync

# Add new dependencies
uv add <package>

# Run Python scripts with uv
uv run python <script.py>
```

### Virtual Environment
The project uses a `.venv` virtual environment that's already created. Use `uv` commands which automatically handle the virtual environment.

## Project Structure

```
project-watch-mcp/
├── src/
│   └── project_watch_mcp/
│       ├── __init__.py           # Package initialization
│       ├── server.py             # MCP server implementation
│       ├── repository_monitor.py # File monitoring with gitignore support
│       └── neo4j_rag.py         # Neo4j RAG system implementation
├── tests/                        # Test files (to be added)
├── pyproject.toml               # Project configuration and dependencies
├── README.md                    # Main documentation
├── CLAUDE.md                    # Claude-specific instructions
├── .gitignore                   # Git ignore patterns
└── .venv/                       # Virtual environment (excluded)

## Development Guidelines

### Key Features to Maintain
1. **Gitignore Support**: Repository monitor must respect `.gitignore` patterns
2. **Real-time Monitoring**: Use `watchfiles` for efficient file system watching
3. **Neo4j Integration**: Maintain vector index support (requires Neo4j 5.11+)
4. **MCP Protocol**: Follow FastMCP patterns for tool implementation

### Testing
```bash
# Run tests
uv run pytest

# With coverage
uv run pytest --cov=src/project_watch_mcp --cov-report=term-missing
```

### Code Quality
```bash
# Format code
uv run black src tests

# Lint code
uv run ruff check src tests

# Type checking
uv run pyright
```

### Adding New MCP Tools
1. Add tool method to `server.py` with proper FastMCP decorators
2. Include comprehensive docstrings for tool documentation
3. Add corresponding tests in `tests/`
4. Update README.md with new tool documentation

### Neo4j Schema
The project uses the following Neo4j structure:
- **Nodes**: `CodeChunk` with properties: file_path, content, chunk_id, language, embeddings
- **Indexes**: Vector index on embeddings for semantic search
- **Relationships**: Can be extended for code relationships (imports, definitions, etc.)

## Common Commands

```bash
# Start the server
uv run project-watch-mcp --repository .

# Run with verbose logging
uv run project-watch-mcp --repository . --verbose

# Run as HTTP server
uv run project-watch-mcp --repository . --transport http --port 8080

# Build for distribution
uv build

# Install locally for testing
uv pip install -e .
```

## Important Notes

- **Neo4j Required**: The project will not function without a running Neo4j instance
- **Python 3.11+**: Uses modern Python features, ensure correct version
- **Gitignore Awareness**: File monitoring automatically respects `.gitignore` patterns
- **Memory Usage**: Large repositories may require Neo4j memory tuning
- **Real-time Updates**: File changes are detected and indexed automatically

## JetBrains MCP Integration

The JetBrains MCP server provides powerful IDE integration capabilities for development. Use these tools to interact with JetBrains IDEs (IntelliJ IDEA, PyCharm, etc.) directly from Claude.

### Available JetBrains MCP Tools

#### File Operations
- `mcp__jetbrains__get_open_in_editor_file_text` - Get content of currently open file
- `mcp__jetbrains__get_open_in_editor_file_path` - Get path of currently open file
- `mcp__jetbrains__get_selected_in_editor_text` - Get selected text from editor
- `mcp__jetbrains__replace_selected_text` - Replace selected text in editor
- `mcp__jetbrains__replace_current_file_text` - Replace entire file content
- `mcp__jetbrains__create_new_file_with_text` - Create new file with content
- `mcp__jetbrains__get_file_text_by_path` - Read file by project path
- `mcp__jetbrains__replace_file_text_by_path` - Replace file content by path
- `mcp__jetbrains__replace_specific_text` - Replace specific text occurrences (preferred for targeted edits)
- `mcp__jetbrains__open_file_in_editor` - Open a file in the IDE editor
- `mcp__jetbrains__get_all_open_file_texts` - Get text of all open files
- `mcp__jetbrains__get_all_open_file_paths` - List all open file paths

#### Search & Navigation
- `mcp__jetbrains__find_files_by_name_substring` - Search files by name pattern
- `mcp__jetbrains__search_in_files_content` - Search text within all project files
- `mcp__jetbrains__list_files_in_folder` - List contents of a directory
- `mcp__jetbrains__list_directory_tree_in_folder` - Get hierarchical directory tree view

#### Code Analysis
- `mcp__jetbrains__get_current_file_errors` - Get errors/warnings in current file
- `mcp__jetbrains__get_project_problems` - Get all project-wide problems
- `mcp__jetbrains__reformat_current_file` - Apply code formatting to current file
- `mcp__jetbrains__reformat_file` - Format specific file by path

#### Version Control
- `mcp__jetbrains__get_project_vcs_status` - Get VCS status (modified/added/deleted files)
- `mcp__jetbrains__find_commit_by_message` - Search commits by message text

#### Debugging
- `mcp__jetbrains__toggle_debugger_breakpoint` - Add/remove breakpoint at line
- `mcp__jetbrains__get_debugger_breakpoints` - List all breakpoints in project

#### Project Management
- `mcp__jetbrains__get_run_configurations` - List available run configurations
- `mcp__jetbrains__run_configuration` - Execute a run configuration
- `mcp__jetbrains__get_project_modules` - List project modules with dependencies
- `mcp__jetbrains__get_project_dependencies` - List all project dependencies

#### IDE Actions
- `mcp__jetbrains__list_available_actions` - List all available IDE actions
- `mcp__jetbrains__execute_action_by_id` - Execute specific IDE action
- `mcp__jetbrains__get_progress_indicators` - Check running background tasks

#### Terminal
- `mcp__jetbrains__get_terminal_text` - Get terminal output
- `mcp__jetbrains__execute_terminal_command` - Run command in IDE terminal

#### Utilities
- `mcp__jetbrains__wait` - Wait for specified milliseconds

### Best Practices for JetBrains MCP

1. **Use for IDE-specific operations**: When working in a JetBrains IDE, prefer these tools over generic file operations for better integration
2. **Leverage code analysis**: Use error detection and project problems tools to identify issues quickly
3. **Prefer `replace_specific_text`**: For targeted edits, this is more efficient than replacing entire files
4. **Check VCS status**: Before making changes, check what files are already modified
5. **Use run configurations**: Execute tests and builds through IDE configurations for consistency
6. **Monitor progress**: Check progress indicators when running long operations

### Example Workflows

```bash
# Quick code review workflow
1. mcp__jetbrains__get_open_in_editor_file_text  # Read current file
2. mcp__jetbrains__get_current_file_errors        # Check for issues
3. mcp__jetbrains__reformat_current_file          # Apply formatting
4. mcp__jetbrains__get_project_vcs_status         # Check what changed

# Search and replace workflow
1. mcp__jetbrains__search_in_files_content        # Find occurrences
2. mcp__jetbrains__replace_specific_text          # Make targeted changes
3. mcp__jetbrains__run_configuration              # Run tests

# Debugging workflow
1. mcp__jetbrains__toggle_debugger_breakpoint     # Set breakpoints
2. mcp__jetbrains__run_configuration              # Run in debug mode
3. mcp__jetbrains__get_terminal_text              # Check output
```