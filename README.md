# Project Watch MCP

A powerful Model Context Protocol (MCP) server that transforms your code repository into an intelligent, searchable knowledge base. Built on Neo4j's graph database, it provides real-time monitoring, AI-powered semantic search, and comprehensive code analysis across multiple programming languages.

## What Can It Do?

- 🔍 **"Find me authentication code"** - Search using natural language, not just keywords
- 📊 **"How complex is this file?"** - Get instant complexity metrics and refactoring suggestions  
- 🚨 **"Show me all TODOs"** - Find patterns across your entire codebase with regex
- 📈 **"What languages are in this repo?"** - Get comprehensive repository statistics
- 🔄 **Real-time updates** - Changes are indexed automatically as you code
- 🎯 **Language-aware** - Understands Python, Java, Kotlin, and 30+ other languages

## Prerequisites

- **Python 3.11+**
- **Neo4j 5.11+** (required for vector index support)
- **uv** package manager (`pip install uv`)
- **OpenAI API key** (optional, for OpenAI embeddings)
- **Local embedding server** (optional, for self-hosted embeddings)

## Quick Start

### 1. Install Neo4j
```bash
# Using Docker (fastest)
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

### 2. Install Project Watch MCP
```bash
# Clone and install
git clone https://github.com/yourusername/project-watch-mcp
cd project-watch-mcp
uv sync
```

### 3. Index Your Repository
```bash
# Initialize and index your repository
uv run project-watch-mcp --initialize --repository /path/to/your/repo
```

### 4. Start the MCP Server
```bash
# For use with Claude Desktop or other MCP clients
uv run project-watch-mcp --repository /path/to/your/repo
```

That's it! Your repository is now indexed and searchable. See the [MCP Tools](#mcp-tools-reference) section for available commands.

## Key Features

### 🔍 Intelligent Code Search
- **Semantic Search**: Find code by meaning, not just keywords - describe what you're looking for in natural language
- **Pattern Matching**: Regex-powered search for specific code patterns, comments, or syntax
- **Language Filtering**: Search within specific programming languages for targeted results

### 📊 Code Intelligence
- **Multi-Language Complexity Analysis**: Analyze Python, Java, and Kotlin code complexity with detailed metrics
- **Maintainability Scoring**: Get actionable insights with maintainability indices and refactoring recommendations
- **Repository Statistics**: Comprehensive metrics on file distribution, languages, and code organization

### 🚀 Real-Time Monitoring
- **Live File Tracking**: Automatically detects and indexes changes as you code
- **Smart Filtering**: Respects `.gitignore` patterns and custom file filters
- **Incremental Updates**: Only re-indexes changed files for optimal performance

### 🧠 Advanced Technology
- **Neo4j Graph Database**: Leverages graph relationships for sophisticated code analysis
- **Native Vector Search**: Direct similarity search without Lucene escaping issues
- **Multiple Embedding Providers**: Choose between OpenAI, Voyage AI, or self-hosted embeddings
- **30+ Language Support**: Indexes and analyzes most common programming languages

### 🛠️ Developer-Friendly
- **8 Powerful MCP Tools**: Complete toolkit for repository analysis and management
- **Multiple Transport Modes**: STDIO for CLI, HTTP for remote access, SSE for streaming
- **Extensive Configuration**: Environment variables and command-line options for full control
- **Battle-Tested**: 1,367+ tests with 85%+ code coverage

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
- `PROJECT_WATCH_USER`: Neo4j username (default: `neo4j`)
- `PROJECT_WATCH_PASSWORD`: Neo4j password (default: `password`)
- `PROJECT_WATCH_DATABASE`: Neo4j database name (default: `neo4j`)
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
export PROJECT_WATCH_USER=neo4j
export PROJECT_WATCH_PASSWORD=mypassword
export PROJECT_WATCH_DATABASE=neo4j

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

## MCP Tools Reference

### 1. `initialize_repository` - Set Up Your Knowledge Base
**Purpose**: Scan and index all repository files for searching and analysis.

**When to Use**:
- First time setup of a new repository
- After major structural changes
- To re-sync if monitoring was interrupted

**Example**:
```python
await initialize_repository()
# Output: {"indexed": 156, "total": 160, "message": "Repository initialized"}
```

**Key Points**:
- Safe to run multiple times (idempotent)
- Automatically starts file monitoring
- Respects .gitignore patterns

### 2. `search_code` - Find Code Intelligently
**Purpose**: Search your repository using natural language or patterns.

**Parameters**:
- `query` (required): What to search for
- `search_type`: "semantic" (default) or "pattern"
- `is_regex`: For pattern search, treat as regex (default: false)
- `limit`: Max results (default: 10, max: 100)
- `language`: Filter by language (e.g., "python", "javascript")

**Examples**:
```python
# Find authentication code semantically
await search_code("user login and password validation")

# Find all TODO comments with regex
await search_code("TODO|FIXME", search_type="pattern", is_regex=True)

# Find Python async functions
await search_code("async functions", language="python")
```

**Understanding Results**:
- Similarity > 0.8 = Highly relevant
- Similarity 0.6-0.8 = Relevant
- Similarity < 0.6 = Loosely related

### 3. `get_repository_stats` - Analyze Your Codebase
**Purpose**: Get comprehensive metrics about your repository.

**Returns**: File counts, language distribution, size metrics, largest files

**Example**:
```python
await get_repository_stats()
# Output: {
#   "total_files": 156,
#   "total_chunks": 1243,
#   "languages": {"python": {"files": 89, "percentage": 57.05}},
#   "largest_files": [{"path": "src/main.py", "size": 45678}]
# }
```

### 4. `get_file_info` - Inspect File Details
**Purpose**: Get metadata about a specific file.

**Example**:
```python
await get_file_info("src/main.py")
# Output: {"path": "src/main.py", "language": "python", "size": 4567, 
#          "lines": 234, "indexed": true, "classes": ["MainApp"]}
```

### 5. `refresh_file` - Update Index for a File
**Purpose**: Force re-indexing of a specific file.

**When to Use**: After external changes, or when monitoring missed an update

**Example**:
```python
await refresh_file("src/updated.py")
# Output: {"status": "success", "action": "updated", "chunks_after": 7}
```

### 6. `delete_file` - Remove from Index
**Purpose**: Remove a file from the index (NOT from filesystem).

**Example**:
```python
await delete_file("src/old.py")
# Output: {"status": "success", "chunks_removed": 12}
```

### 7. `analyze_complexity` - Measure Code Quality
**Purpose**: Analyze code complexity for Python, Java, and Kotlin files.

**Parameters**:
- `file_path` (required): File to analyze
- `include_metrics`: Include maintainability index (default: true)

**Example**:
```python
await analyze_complexity("src/main.py")
# Output: {
#   "summary": {"total_complexity": 42, "maintainability_index": 65.4, "grade": "B"},
#   "functions": [{"name": "process_data", "complexity": 15, "rank": "D"}],
#   "recommendations": ["Consider refactoring 'process_data'"]
# }
```

**Complexity Grades**:
- **A (1-5)**: Simple, maintainable
- **B (6-10)**: Moderate
- **C (11-20)**: Complex, consider refactoring
- **D (21-30)**: Very complex, needs refactoring
- **E-F (31+)**: Urgent refactoring needed

### 8. `monitoring_status` - Check System Health
**Purpose**: Get current monitoring status and pending changes.

**Example**:
```python
await monitoring_status()
# Output: {
#   "is_running": true,
#   "repository_path": "/path/to/repo",
#   "pending_changes": 3,
#   "recent_changes": [{"type": "modified", "path": "src/main.py"}]
# }

## Practical Examples

### Finding and Fixing Complex Code
```python
# 1. Search for authentication logic
results = await search_code("user authentication password validation")

# 2. Analyze complexity of the found files
for result in results[:3]:
    complexity = await analyze_complexity(result["file"])
    if complexity["summary"]["average_complexity"] > 10:
        print(f"⚠️ {result['file']} needs refactoring")
        print(f"   Recommendations: {complexity['recommendations']}")
```

### Tracking Code Patterns
```python
# Find all TODO comments across the codebase
todos = await search_code("TODO|FIXME|HACK", search_type="pattern", is_regex=True)
print(f"Found {len(todos)} action items to address")

# Find all error handling patterns
error_handlers = await search_code("try except error handling", language="python")
```

### Repository Health Check
```python
# Get overall repository metrics
stats = await get_repository_stats()
print(f"Repository contains {stats['total_files']} files")
print(f"Primary language: {max(stats['languages'], key=lambda x: stats['languages'][x]['percentage'])}")

# Check monitoring health
status = await monitoring_status()
if status["pending_changes"] > 10:
    print("⚠️ High number of pending changes - indexing may be delayed")
```

## Common Use Cases

### 1. Code Review Assistance
- Find similar implementations: `await search_code("functionality description")`
- Check complexity before merge: `await analyze_complexity("new_feature.py")`
- Verify pattern consistency: `await search_code("logging pattern", language="python")`

### 2. Refactoring Support
- Identify complex areas: Loop through files and run `analyze_complexity()`
- Find duplicate patterns: Search for similar code semantically
- Track refactoring progress: Use `get_repository_stats()` before and after

### 3. Onboarding New Developers
- Explore codebase: `await search_code("main entry point")`
- Understand architecture: `await search_code("database connection setup")`
- Find examples: `await search_code("how to use API client")`

### 4. Documentation Generation
- Find undocumented functions: `await search_code("def.*:\n[^#]", search_type="pattern", is_regex=True)`
- Identify public APIs: `await search_code("public interface exported")`
- Track documentation coverage: Compare file counts with documented files

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

## Troubleshooting Guide

### 🔴 Neo4j Connection Errors
```bash
Error: Unable to connect to Neo4j at bolt://localhost:7687
```
**Solutions**:
1. Check Neo4j is running: on Neo4j Desktop
2. Verify credentials: default is `neo4j/password`
3. Test connection: Visit http://localhost:7474 in browser
4. Check ports: Ensure 7687 and 7474 are not blocked

### 🔴 Files Not Being Indexed
```bash
Indexed 0/156 files
```
**Solutions**:
1. Check file patterns: `--file-patterns "*.py,*.js"`
2. Verify permissions: `ls -la /path/to/repo`
3. Check .gitignore: Ensure files aren't excluded
4. Run with verbose: `--verbose` for detailed logs

### 🔴 Embedding Errors
```bash
Failed to generate embeddings: API key invalid
```
**Solutions**:
- **OpenAI**: Verify `OPENAI_API_KEY` is set and valid
- **Local**: Check server at `LOCAL_EMBEDDING_API_URL`
- **Fallback**: Use `EMBEDDING_PROVIDER=mock` for testing

### 🔴 High Memory Usage
**For large repositories (>10,000 files)**:
1. Limit file patterns: Focus on specific languages
2. Increase Neo4j heap: Edit `neo4j.conf`
3. Use batch mode: `--initialize` then start server
4. Consider chunking: Index in stages

### 🟡 Performance Tips
- **Initial indexing**: ~100 files/minute with OpenAI embeddings
- **Search latency**: <100ms for most queries
- **Memory usage**: ~1GB for 10,000 files
- **CPU usage**: <5% during monitoring

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
  --neo4j-password "$PROJECT_WATCH_PASSWORD" \
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

## Testing

Project Watch MCP has comprehensive test coverage with 1,367+ tests ensuring reliability:

### Running Tests
```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=project_watch_mcp --cov-report=html

# Run specific test categories
uv run pytest -m unit          # Fast unit tests
uv run pytest -m integration   # Integration tests
uv run pytest -m "not slow"    # Skip slow tests
```

### Test Coverage
- **Overall**: 85%+ line coverage
- **Critical paths**: 90%+ coverage
- **Complexity analysis**: 100% coverage
- **MCP tools**: 85% coverage

### Contributing
When contributing, please ensure:
1. All tests pass: `uv run pytest`
2. Code is formatted: `uv run black src tests`
3. Code is linted: `uv run ruff check src tests`
4. New features include tests

## License

MIT