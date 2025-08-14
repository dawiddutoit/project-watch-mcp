# Project Watch MCP

Advanced repository monitoring MCP server with Neo4j-based RAG system, featuring native vector search, multi-language detection, and language-specific complexity analysis.

## Prerequisites

- **Python 3.11+**
- **Neo4j 5.11+** (required for vector index support)
- **uv** package manager (`pip install uv`)
- **OpenAI API key** (optional, for OpenAI embeddings)
- **Local embedding server** (optional, for self-hosted embeddings)

## Features

### Core Features
- **Real-time Repository Monitoring**: Uses `watchfiles` library to detect file changes
- **Neo4j-based RAG System**: Stores code chunks with embeddings for semantic search
- **FastMCP Server**: Provides MCP tools for querying the repository knowledge base
- **Gitignore Support**: Automatically respects `.gitignore` patterns to exclude files from monitoring

### Advanced Features (New!)
- **Neo4j Native Vector Search**: Eliminates Lucene escaping issues with direct vector similarity
- **Multi-Language Detection**: Hybrid detection using tree-sitter, Pygments, and file extensions
- **Language-Specific Complexity Analysis**: Cyclomatic and cognitive complexity for Python, Java, and Kotlin
- **Embedding Enrichment**: Language-aware embedding adjustments for better search accuracy
- **Performance Caching**: Efficient caching layers for detection and analysis results

See [Advanced Features Documentation](docs/advanced_features.md) for detailed information.

## Quick Neo4j Setup

### Using Neo4j Desktop (Recommended)
Download and install Neo4j Desktop from https://neo4j.com/download/
- Create a new project and database
- Start the database
- Default connection will be at neo4j://localhost:7687

### Using Docker (Alternative)
```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

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

### Neo4j Configuration
Set the following environment variables:

- `NEO4J_URI`: Neo4j connection URI (default: `bolt://localhost:7687`)
- `NEO4J_USER`: Neo4j username (default: `neo4j`)
- `NEO4J_PASSWORD`: Neo4j password (default: `password`)
- `NEO4J_DATABASE`: Neo4j database name (default: `neo4j`)
- `REPOSITORY_PATH`: Path to repository to monitor (default: current directory)
- `FILE_PATTERNS`: Comma-separated file patterns to monitor (default: common code files)

### Advanced Configuration

#### Neo4j Native Vector Search
Configure Neo4j to use native vector indexes instead of Lucene:

```bash
# Enable native vector search
export NEO4J_VECTOR_INDEX_ENABLED=true
export VECTOR_SIMILARITY_METRIC=cosine  # or euclidean
```

#### Language Detection
Configure the hybrid language detection system:

```bash
# Language detection settings
export LANGUAGE_DETECTION_CACHE_SIZE=1000
export LANGUAGE_DETECTION_CACHE_TTL=3600
export LANGUAGE_CONFIDENCE_THRESHOLD=0.85
```

#### Complexity Analysis
Configure complexity thresholds and analysis behavior:

```bash
# Complexity thresholds
export COMPLEXITY_THRESHOLD_HIGH=10
export COMPLEXITY_THRESHOLD_VERY_HIGH=20
export COMPLEXITY_INCLUDE_METRICS=true
```

### Embedding Provider Configuration

The system supports multiple embedding providers for semantic search:

#### 1. Mock Provider (Default)
For testing and development without external dependencies:
```bash
export EMBEDDING_PROVIDER=mock
export EMBEDDING_DIMENSION=384  # Optional, default is 384
```

#### 2. OpenAI Provider
For high-quality semantic search using OpenAI's embedding models:
```bash
export EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=your-api-key-here
export OPENAI_EMBEDDING_MODEL=text-embedding-3-small  # Optional, default
```

Available OpenAI models:
- `text-embedding-3-small` (1536 dimensions) - Recommended
- `text-embedding-3-large` (3072 dimensions) - Higher quality
- `text-embedding-ada-002` (1536 dimensions) - Legacy

#### 3. Local Provider
For self-hosted embedding servers:
```bash
export EMBEDDING_PROVIDER=local
export LOCAL_EMBEDDING_API_URL=http://localhost:8080/embeddings
export EMBEDDING_DIMENSION=384  # Must match your model's dimension
```

Your local API should accept POST requests:
```json
# Single embedding
POST /embeddings
{"text": "code to embed"}
→ {"embedding": [0.1, 0.2, ...]}

# Batch embeddings
POST /embeddings
{"texts": ["code 1", "code 2"]}
→ {"embeddings": [[0.1, ...], [0.3, ...]]}
```

### File Filtering

The repository monitor automatically uses the project's `.gitignore` file to determine which files to exclude from monitoring. This ensures that:
- Build artifacts, dependencies, and temporary files are ignored
- Version control directories (.git) are excluded
- Virtual environments and cache directories are skipped
- Any custom patterns in your `.gitignore` are respected

If no `.gitignore` file exists, the monitor falls back to sensible defaults (excluding common directories like `node_modules`, `.venv`, `__pycache__`, etc.).

## Quick Start with Advanced Features

### Example: Find Complex Authentication Code

```python
# Initialize with all advanced features
await initialize_repository()

# Search for authentication code using native vectors
results = await search_code(
    query="user authentication JWT validation",
    search_type="semantic",
    language="python"  # Language-filtered search
)

# Analyze complexity of found files
for result in results[:5]:
    complexity = await analyze_complexity(
        file_path=result["file"],
        include_metrics=True
    )
    
    print(f"\nFile: {result['file']}")
    print(f"  Similarity: {result['similarity']:.2f}")
    print(f"  Complexity Grade: {complexity['summary']['complexity_grade']}")
    print(f"  Maintainability: {complexity['summary']['maintainability_index']:.1f}")
    
    # Show complex functions
    complex_funcs = [
        f for f in complexity['functions'] 
        if f['complexity'] > 10
    ]
    if complex_funcs:
        print(f"  Complex functions: {', '.join(f['name'] for f in complex_funcs)}")
```

## Usage Examples

### Repository Initialization

The `--initialize` flag allows you to index your repository directly from the command line without starting the MCP server. This is useful for:
- Pre-indexing repositories before using MCP clients
- Batch processing multiple repositories
- CI/CD pipelines
- Testing and debugging indexing issues

#### Basic Initialization
```bash
# Initialize the current directory
uv run project-watch-mcp --initialize

# Initialize a specific repository
uv run project-watch-mcp --initialize --repository /path/to/repo
```

#### Initialization with Verbose Output
```bash
# See detailed progress during initialization
uv run project-watch-mcp --initialize --verbose
```

#### Initialization with Custom Configuration
```bash
# Initialize with specific Neo4j connection
uv run project-watch-mcp --initialize \
  --repository /path/to/repo \
  --neo4j-uri bolt://myserver:7687 \
  --neo4j-password mypassword \
  --verbose

# Initialize with custom project name
uv run project-watch-mcp --initialize \
  --repository /path/to/repo \
  --project-name my-awesome-project \
  --verbose

# Initialize with specific file patterns
uv run project-watch-mcp --initialize \
  --repository /path/to/repo \
  --file-patterns "*.py,*.js,*.ts" \
  --verbose
```

#### When to Use CLI Initialization vs MCP Tool

**Use `--initialize` CLI flag when:**
- Setting up a new repository for the first time
- Running batch initialization for multiple repositories
- Integrating with CI/CD pipelines
- Debugging indexing issues (with `--verbose`)
- Pre-indexing before starting the MCP server

**Use `initialize_repository` MCP tool when:**
- Already connected through an MCP client
- Need to re-index after significant changes
- Part of an automated workflow within Claude or another MCP client
- Want to trigger initialization programmatically

### Basic Usage (STDIO Mode for MCP Clients)
```bash
# Monitor current repository (without initialization)
project-watch-mcp --repository .

# Monitor specific repository (without initialization)
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
# Neo4j configuration
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=mypassword
export NEO4J_DATABASE=neo4j

# Repository configuration
export REPOSITORY_PATH=/path/to/repo
export FILE_PATTERNS="*.py,*.js,*.ts"

# Embedding configuration (choose one)
# For OpenAI:
export EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=sk-...

# For Local:
export EMBEDDING_PROVIDER=local
export LOCAL_EMBEDDING_API_URL=http://localhost:8080/embeddings

# For Mock (default):
export EMBEDDING_PROVIDER=mock

project-watch-mcp
```

## Command-Line Options

```
--initialize             Initialize repository index and exit (no server)
--neo4j-uri URI           Neo4j connection URI (default: bolt://localhost:7687)
--neo4j-user USER         Neo4j username (default: neo4j)
--neo4j-password PASS     Neo4j password (default: password)
--neo4j-database DB       Neo4j database name (default: neo4j)
--repository, -r PATH     Path to repository to monitor (default: current directory)
--project-name NAME       Custom project name for Neo4j (default: derived from path)
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
        "NEO4J_URI": "bolt://localhost:7687",
        "EMBEDDING_PROVIDER": "openai",
        "OPENAI_API_KEY": "your-openai-api-key"
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

### 1. initialize_repository
Scan and index all files in the repository for semantic search.

**Key Features:**
- Idempotent operation (safe to run multiple times)
- Respects .gitignore patterns automatically
- Starts real-time file monitoring after indexing
- Supports 30+ programming languages and file types

**Example Usage:**
```python
# First-time initialization
await initialize_repository()
# Returns: {"indexed": 42, "total": 45}

# Re-initialization updates only changed files
await initialize_repository()
# Returns: {"indexed": 3, "total": 45}
```

### 2. search_code
Search repository using AI-powered semantic search or pattern matching.

**Search Types:**
- **Semantic**: Find conceptually similar code using AI embeddings
- **Pattern**: Find exact text matches or regex patterns

**Example Usage:**
```python
# Semantic search - find authentication logic
await search_code(
    query="user authentication and JWT token validation",
    search_type="semantic",
    limit=5
)

# Pattern search with regex - find TODO comments
await search_code(
    query="TODO|FIXME|HACK",
    search_type="pattern",
    is_regex=True,
    limit=10
)

# Language-specific search
await search_code(
    query="async function implementations",
    language="typescript"
)
```

**Similarity Scores:**
- `> 0.8`: Very relevant
- `0.6-0.8`: Relevant
- `< 0.6`: Loosely related

### 3. get_repository_stats
Get comprehensive statistics about the indexed repository.

**Returns:**
- Total files, chunks, size, and lines
- Language breakdown with percentages
- Largest files in the repository
- Index health and coverage metrics

**Example Output:**
```json
{
    "total_files": 156,
    "total_chunks": 1243,
    "languages": {
        "python": {"files": 89, "percentage": 57.05},
        "javascript": {"files": 45, "percentage": 28.85}
    }
}
```

### 4. get_file_info
Get detailed metadata about a specific file.

**Accepts:**
- Relative paths from repository root
- Absolute paths within repository

**Returns:**
- File path, size, language, and modification time
- Indexing status and chunk count
- Extracted code elements (imports, classes, functions)

**Example Usage:**
```python
# Using relative path
await get_file_info("src/main.py")

# Using absolute path
await get_file_info("/home/user/project/README.md")
```

### 5. refresh_file
Manually refresh a specific file in the index.

**Use Cases:**
- Force immediate re-indexing after changes
- Add newly created files to index
- Update index when automatic monitoring missed changes

**Example Usage:**
```python
await refresh_file("src/updated_module.py")
# Returns: {"status": "success", "action": "updated", "chunks_after": 7}
```

### 6. delete_file
Remove a file and its chunks from the Neo4j index.

**Important:** This only removes the file from the index - it does NOT delete the actual file from the filesystem.

**Use Cases:**
- Clean up index after files are deleted from repository
- Remove temporarily indexed files
- Clear outdated or moved files from index

**Example Usage:**
```python
# Delete a file from the index
await delete_file("src/deprecated_module.py")
# Returns: {"status": "success", "chunks_removed": 12}

# File not in index
await delete_file("non_indexed_file.py")
# Returns: {"status": "warning", "message": "File not found in index"}
```

### 7. analyze_complexity
Calculate code complexity with language-specific understanding for Python, Java, and Kotlin.

**Key Features:**
- **Multi-language support**: Python, Java, and Kotlin analyzers
- **Dual metrics**: Both cyclomatic and cognitive complexity
- **Maintainability Index**: Overall code health score (0-100)
- **Language-specific features**: Handles unique constructs per language
- **Smart recommendations**: Actionable refactoring suggestions

**Parameters:**
- `file_path`: Path to the code file to analyze
- `include_metrics`: Include additional metrics like maintainability index (default: true)

**Example:**
```json
{
  "file": "src/complex_module.py",
  "summary": {
    "total_complexity": 42,
    "average_complexity": 5.2,
    "maintainability_index": 65.4,
    "complexity_grade": "B"
  },
  "functions": [
    {
      "name": "process_data",
      "complexity": 15,
      "rank": "D",
      "line": 45,
      "classification": "complex"
    }
  ],
  "recommendations": [
    "Consider refactoring 'process_data' (complexity: 15)",
    "3 functions have complexity > 10"
  ]
}
```

**Complexity Rankings:**
- **A (1-5)**: Simple, easy to understand
- **B (6-10)**: Moderate complexity
- **C (11-20)**: Complex, consider refactoring
- **D (21-30)**: Very complex, needs refactoring
- **E (31-40)**: High risk, difficult to maintain
- **F (41+)**: Extremely complex, urgent refactoring needed

### 8. monitoring_status
Check the current repository monitoring status.

**Returns:**
- Monitoring state (running/stopped)
- Repository path and file patterns
- Pending changes queue
- Recent file changes with timestamps
- Monitoring statistics

**Change Types:**
- `added`: New file created
- `modified`: Existing file changed
- `deleted`: File removed

## Architecture

The system is built with a modular architecture consisting of five main components:

1. **Core Module** (`src/project_watch_mcp/core/`): Shared business logic
   - `neo4j_rag.py`: Neo4j database operations and RAG functionality
   - `repository_monitor.py`: File system monitoring with gitignore support
   - `__init__.py`: Module exports and initialization

2. **Repository Monitor**: Watches for file changes using `watchfiles`
   - Automatically loads and respects `.gitignore` patterns
   - Falls back to sensible defaults if no `.gitignore` exists
   - Supports additional custom ignore patterns

3. **Neo4j RAG**: Manages code indexing and retrieval with Neo4j
   - Vector indexing for semantic search
   - Code chunking and metadata extraction
   - Efficient graph-based storage and retrieval

4. **Embedding Provider**: Generates vector representations for semantic search
   - OpenAI: Cloud-based, high-quality embeddings
   - Local: Self-hosted embedding server
   - Mock: Deterministic embeddings for testing

5. **MCP Server**: Exposes functionality through MCP tools
   - FastMCP-based implementation
   - Supports STDIO, HTTP, and SSE transports
   - Provides 8 specialized tools for code analysis

The modular design allows the core functionality to be used both through the MCP server and directly via the CLI, enabling flexible integration patterns.

## Troubleshooting

### Neo4j Connection Issues
If you get connection errors:
1. Ensure Neo4j is running: Check Neo4j Desktop or `docker ps` if using Docker
2. Verify the browser is accessible at http://localhost:7474
3. Verify credentials match your Neo4j setup
4. Check firewall settings for ports 7474 and 7687

### File Monitoring Issues
If files aren't being detected:
1. Check file patterns match your files
2. Ensure you have read permissions for the repository
3. Use `--verbose` flag for detailed logging
4. Verify `.gitignore` patterns aren't excluding desired files

### Performance Considerations
- Initial indexing time depends on repository size and embedding provider
- Large repositories (>10,000 files) may take several minutes
- Consider using specific file patterns to reduce scope
- Neo4j memory settings may need adjustment for large codebases
- Embedding performance:
  - Mock: Instant (no real embeddings)
  - OpenAI: ~100-500ms per embedding (API calls)
  - Local: Depends on your hardware and model

### Embedding Provider Issues
If embeddings fail:
1. **OpenAI**: Check API key is valid and has credits
2. **Local**: Ensure embedding server is running at configured URL
3. **Mock**: Should always work (for testing only)
4. Use `--verbose` flag to see detailed error messages

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