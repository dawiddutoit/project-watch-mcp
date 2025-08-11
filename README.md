# Project Watch

Repository monitoring MCP server with Neo4j-based RAG capabilities.

## Requirements

- **Python 3.11+**
- **Neo4j 5.11+** (required for vector index support)
- **uv** package manager

## Features

- **Real-time Repository Monitoring**: Uses `watchfiles` library to detect file changes
- **Neo4j-based RAG System**: Stores code chunks with embeddings for semantic search
- **FastMCP Server**: Provides MCP tools for querying the repository knowledge base
- **Multi-language Support**: Automatically detects and indexes various programming languages
- **Semantic Search**: Find code by meaning, not just text matching
- **Pattern Search**: Support for regex and text-based pattern matching

## Installation

```bash
uv sync
```

## Configuration

Set the following environment variables:

- `NEO4J_URI`: Neo4j connection URI (default: `bolt://localhost:7687`)
- `NEO4J_USER`: Neo4j username (default: `neo4j`)
- `NEO4J_PASSWORD`: Neo4j password (default: `password`)
- `NEO4J_DATABASE`: Neo4j database name (default: `neo4j`)
- `REPOSITORY_PATH`: Path to repository to monitor (default: current directory)
- `FILE_PATTERNS`: Comma-separated file patterns to monitor (default: common code files)

## Usage

### Running as MCP Server (STDIO)

```bash
uv run project-watch-mcp-mcp --repository /path/to/repo
```

### Running as HTTP Server

```bash
uv run project-watch-mcp-mcp --repository /path/to/repo --transport http --port 8000
```

## MCP Tools

The server provides the following MCP tools:

1. **initialize_repository**: Scan and index all files in the repository
2. **search_code**: Search for code using semantic or pattern matching
3. **get_repository_stats**: Get statistics about the indexed repository
4. **get_file_info**: Get metadata about a specific file
5. **refresh_file**: Manually refresh a file in the index
6. **monitoring_status**: Check the current monitoring status

## Architecture

The system consists of three main components:

1. **Repository Monitor**: Watches for file changes using `watchfiles`
2. **Neo4j RAG**: Manages code indexing and retrieval with Neo4j
3. **MCP Server**: Exposes functionality through MCP tools

## Development

### Running Tests

```bash
uv run pytest
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

## License

MIT