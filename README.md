# Project Watch MCP

Repository monitoring MCP server that creates a Neo4j-based RAG (Retrieval-Augmented Generation) system for your codebase with intelligent file filtering.

## Prerequisites

- **Python 3.11+**
- **Neo4j 5.11+** (required for vector index support)
- **uv** package manager (`pip install uv`)

## Features

- **Real-time Repository Monitoring**: Uses `watchfiles` library to detect file changes
- **Neo4j-based RAG System**: Stores code chunks with embeddings for semantic search
- **FastMCP Server**: Provides MCP tools for querying the repository knowledge base
- **Multi-language Support**: Automatically detects and indexes various programming languages
- **Semantic Search**: Find code by meaning, not just text matching
- **Pattern Search**: Support for regex and text-based pattern matching
- **Gitignore Support**: Automatically respects `.gitignore` patterns to exclude files from monitoring

## Quick Neo4j Setup

### Using Docker (Recommended)
```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

### Or Install Locally
Download from https://neo4j.com/download/

## Installation

### Option 1: Install from PyPI (Coming Soon)
```bash
uvx project-watch-mcp --repository /path/to/your/repo
```

### Option 2: Install from Local Package
```bash
# Clone and build
git clone <repository-url>
cd project-watch-mcp
uv build

# Run from built wheel
uvx --from ./dist/project-watch-mcp-0.1.0-py3-none-any.whl project-watch-mcp --repository /path/to/repo
```

### Option 3: Install Globally
```bash
# From the project directory
uv pip install .

# Then run from anywhere
project-watch-mcp --repository /path/to/repo
```

### Option 4: Development Installation
```bash
# Clone the repository
git clone <repository-url>
cd project-watch-mcp

# Install dependencies
uv sync

# Run directly
uv run project-watch-mcp --repository /path/to/repo
```

## Configuration

Set the following environment variables:

- `NEO4J_URI`: Neo4j connection URI (default: `bolt://localhost:7687`)
- `NEO4J_USER`: Neo4j username (default: `neo4j`)
- `NEO4J_PASSWORD`: Neo4j password (default: `password`)
- `NEO4J_DATABASE`: Neo4j database name (default: `neo4j`)
- `REPOSITORY_PATH`: Path to repository to monitor (default: current directory)
- `FILE_PATTERNS`: Comma-separated file patterns to monitor (default: common code files)

### File Filtering

The repository monitor automatically uses the project's `.gitignore` file to determine which files to exclude from monitoring. This ensures that:
- Build artifacts, dependencies, and temporary files are ignored
- Version control directories (.git) are excluded
- Virtual environments and cache directories are skipped
- Any custom patterns in your `.gitignore` are respected

If no `.gitignore` file exists, the monitor falls back to sensible defaults (excluding common directories like `node_modules`, `.venv`, `__pycache__`, etc.).

## Usage Examples

### Basic Usage (STDIO Mode for MCP Clients)
```bash
# Monitor current repository
project-watch-mcp --repository .

# Monitor specific repository
project-watch-mcp --repository /path/to/repo
```

### HTTP Server Mode
```bash
# Run as HTTP server for remote access
project-watch-mcp \
  --repository /path/to/repo \
  --transport http \
  --port 8080
```

### Custom Neo4j Connection
```bash
project-watch-mcp \
  --repository /path/to/repo \
  --neo4j-uri bolt://myserver:7687 \
  --neo4j-user myuser \
  --neo4j-password mypassword \
  --neo4j-database mydb
```

### Monitor Specific File Types
```bash
project-watch-mcp \
  --repository /path/to/repo \
  --file-patterns "*.py,*.js,*.ts,*.jsx,*.tsx"
```

### Using Environment Variables
```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=mypassword
export NEO4J_DATABASE=neo4j
export REPOSITORY_PATH=/path/to/repo
export FILE_PATTERNS="*.py,*.js,*.ts"

project-watch-mcp
```

## Command-Line Options

```
--neo4j-uri URI           Neo4j connection URI (default: bolt://localhost:7687)
--neo4j-user USER         Neo4j username (default: neo4j)
--neo4j-password PASS     Neo4j password (default: password)
--neo4j-database DB       Neo4j database name (default: neo4j)
--repository, -r PATH     Path to repository to monitor (required)
--file-patterns, -p PATS  Comma-separated file patterns (default: common code files)
--transport, -t TYPE      Transport type: stdio, http, sse (default: stdio)
--host HOST              HTTP/SSE server host (default: 127.0.0.1)
--port PORT              HTTP/SSE server port (default: 8000)
--path PATH              HTTP/SSE server path (default: /mcp/)
--verbose, -v            Enable verbose logging
--help, -h               Show help message
```

## MCP Client Configuration

### Claude Desktop
Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "project-watch": {
      "command": "uvx",
      "args": [
        "project-watch-mcp",
        "--repository",
        "/path/to/your/repo",
        "--neo4j-password",
        "your-password"
      ],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687"
      }
    }
  }
}
```

### Other MCP Clients
The server supports three transport modes:
- **stdio** (default): For direct MCP client integration
- **http**: REST API at `http://host:port/mcp/`
- **sse**: Server-Sent Events for streaming

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
   - Automatically loads and respects `.gitignore` patterns
   - Falls back to sensible defaults if no `.gitignore` exists
   - Supports additional custom ignore patterns
2. **Neo4j RAG**: Manages code indexing and retrieval with Neo4j
3. **MCP Server**: Exposes functionality through MCP tools

## Troubleshooting

### Neo4j Connection Issues
If you get connection errors:
1. Ensure Neo4j is running: `docker ps` or check http://localhost:7474
2. Verify credentials match your Neo4j setup
3. Check firewall settings for ports 7474 and 7687

### File Monitoring Issues
If files aren't being detected:
1. Check file patterns match your files
2. Ensure you have read permissions for the repository
3. Use `--verbose` flag for detailed logging
4. Verify `.gitignore` patterns aren't excluding desired files

### Performance Considerations
- Initial indexing time depends on repository size
- Large repositories (>10,000 files) may take several minutes
- Consider using specific file patterns to reduce scope
- Neo4j memory settings may need adjustment for large codebases

## Advanced Usage

### Docker Compose Setup
Create a `docker-compose.yml`:

```yaml
version: '3.8'
services:
  neo4j:
    image: neo4j:latest
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password
    volumes:
      - neo4j_data:/data

volumes:
  neo4j_data:
```

Then run:
```bash
docker-compose up -d
project-watch-mcp --repository . --neo4j-password password
```

### Running from Another Project
Once published to PyPI, you can use `uvx` to run Project Watch MCP from any project:

```bash
# Direct usage with uvx (no installation needed)
uvx project-watch-mcp --repository . --neo4j-password your-password

# Or create a shell script for convenience
#!/bin/bash
# run-project-watch.sh

uvx project-watch-mcp \
  --repository "$(pwd)" \
  --neo4j-password "$NEO4J_PASSWORD" \
  "$@"
```

For local development before PyPI publication:
```bash
# Using local package
uvx --from /path/to/project-watch-mcp project-watch-mcp --repository .
```

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

### Development Installation
```bash
# Install in development mode
uv sync
uv pip install -e .

# Run tests
uv run pytest

# Format and lint
uv run black .
uv run ruff check .
```

## License

MIT