# Project Watch - UVX Installation & Usage Guide

Project Watch is a repository monitoring MCP server that creates a Neo4j-based RAG (Retrieval-Augmented Generation) system for your codebase. It can be easily installed and run using `uvx` from any repository.

## Prerequisites

1. **Python 3.11+** installed
2. **uv** package manager installed (`pip install uv`)
3. **Neo4j 5.11+** database running (required for vector index support)

### Quick Neo4j Setup

Using Docker (recommended):
```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

Or install locally from https://neo4j.com/download/

## Installation Options

### Option 1: Run Directly from GitHub (Coming Soon)

Once published to PyPI:
```bash
uvx project-watch-mcp-mcp --repository /path/to/your/repo
```

### Option 2: Run from Local Package

From the project-watch-mcp-mcp directory:
```bash
# Build the package
uv build

# Run from the built wheel
uvx --from ./dist/project-watch-mcp-mcp-0.1.0-py3-none-any.whl project-watch-mcp-mcp --repository /path/to/repo
```

### Option 3: Install Globally with UV

```bash
# From the project-watch-mcp-mcp directory
uv pip install .

# Then run from anywhere
project-watch-mcp-mcp --repository /path/to/repo
```

## Usage Examples

### Basic Usage (STDIO Mode for MCP Clients)

Monitor current repository with default settings:
```bash
uvx project-watch-mcp-mcp --repository .
```

### HTTP Server Mode

Run as an HTTP server for remote access:
```bash
uvx project-watch-mcp-mcp \
  --repository /path/to/repo \
  --transport http \
  --port 8080
```

### Custom Neo4j Connection

```bash
uvx project-watch-mcp-mcp \
  --repository /path/to/repo \
  --neo4j-uri bolt://myserver:7687 \
  --neo4j-user myuser \
  --neo4j-password mypassword \
  --neo4j-database mydb
```

### Monitor Specific File Types

```bash
uvx project-watch-mcp-mcp \
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
export MCP_TRANSPORT=stdio

uvx project-watch-mcp-mcp
```

## MCP Client Configuration

### For Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "project-watch-mcp": {
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

### For Other MCP Clients

The server supports three transport modes:
- **stdio** (default): For direct MCP client integration
- **http**: REST API at `http://host:port/mcp/`
- **sse**: Server-Sent Events for streaming

## Available MCP Tools

Once running, the server provides these MCP tools:

1. **initialize_repository**: Scan and index the repository
2. **search_code**: Semantic or pattern-based code search
3. **get_repository_stats**: Get repository statistics
4. **get_file_info**: Get metadata about specific files
5. **refresh_file**: Manually refresh a file in the index
6. **monitoring_status**: Check file monitoring status

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

### Performance Considerations

- Initial indexing time depends on repository size
- Large repositories (>10,000 files) may take several minutes
- Consider using specific file patterns to reduce scope
- Neo4j memory settings may need adjustment for large codebases

## Development & Contributing

To develop or modify project-watch-mcp-mcp:

```bash
# Clone the repository
git clone <repository-url>
cd project-watch-mcp-mcp

# Install in development mode
uv sync
uv pip install -e .

# Run tests
uv run pytest

# Format and lint
uv run black .
uv run ruff check .
```

## Advanced Usage

### Running from Another Project

Create a shell script in your project:

```bash
#!/bin/bash
# run-project-watch-mcp-mcp.sh

uvx --from /path/to/project-watch-mcp-mcp/dist/project-watch-mcp-mcp-0.1.0-py3-none-any.whl \
  project-watch-mcp-mcp \
  --repository "$(pwd)" \
  --neo4j-password "$NEO4J_PASSWORD" \
  "$@"
```

Then run: `./run-project-watch-mcp-mcp.sh`

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

Then:
```bash
docker-compose up -d
uvx project-watch-mcp-mcp --repository . --neo4j-password password
```

## License

MIT License - See LICENSE file for details