# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Project Watch MCP is a repository monitoring MCP server that creates a Neo4j-based RAG (Retrieval-Augmented Generation) system for codebases. It provides real-time file monitoring, semantic code search, and intelligent indexing capabilities through the Model Context Protocol (MCP).

**Version**: 0.1.0  
**Status**: Active development  
**Python**: 3.11+  
**Main Dependencies**: Neo4j 5.11+, FastMCP, watchfiles

## Development Setup

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