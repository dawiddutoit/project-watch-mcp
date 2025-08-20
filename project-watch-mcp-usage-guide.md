# Project Watch MCP Testing Guide

This guide provides comprehensive instructions for testing the Project Watch MCP server tools. Use this to verify all features are working correctly and to provide feedback on the tool's functionality.

## <� Quick Start Testing Commands

Copy and paste these exact commands to test the MCP tools:

```bash
# Initialize repository monitoring
Use mcp__project-watch-mcp__initialize_repository

# Check indexing status
Use mcp__project-watch-mcp__indexing_status

# Search for test files semantically
Use mcp__project-watch-mcp__search_code with query="test files" and search_type="semantic"

# Get repository statistics
Use mcp__project-watch-mcp__get_repository_stats

# Search for test files using file classification
Use mcp__project-watch-mcp__search_code with query="test" and file_category="test"

# List all configuration files
Use mcp__project-watch-mcp__list_files_by_category with category="config"
```

## =� Testing Prerequisites

### 1. Verify Neo4j is Running

```bash
# Check Neo4j connection (should return without error)
Use mcp__project-watch-mcp__monitoring_status

# Expected output if Neo4j is NOT running:
# Error: Unable to connect to Neo4j at bolt://localhost:7687
```

### 2. Check MCP Server Status

```bash
# Verify the MCP server is available
Use mcp__project-watch-mcp__monitoring_status

# Expected successful output:
{
  "is_running": true|false,
  "repository_path": "/path/to/repo",
  "version_info": {...}
}
```

### 3. Verify Embedding Provider

```bash
# Initialize to test embedding provider
Use mcp__project-watch-mcp__initialize_repository

# If using OpenAI, errors will mention API key issues
# If using mock provider, should work immediately
```

## >� Systematic Test Scenarios

### Test 1: Repository Initialization

```bash
# Step 1: Initialize the repository
Use mcp__project-watch-mcp__initialize_repository

# Expected output:
{
  "status": "indexing_started",
  "total": <number>,
  "message": "Background indexing started for X files."
}

# Step 2: Check indexing progress
Use mcp__project-watch-mcp__indexing_status

# Expected output while indexing:
{
  "status": "in_progress",
  "indexed_files": <number>,
  "total_estimate": <number>
}

# Step 3: Wait and check again until complete
Use mcp__project-watch-mcp__indexing_status

# Expected output when complete:
{
  "status": "completed",
  "indexed_files": <number>,
  "duration_seconds": <number>
}
```

### Test 2: Semantic Search

```bash
# Test 2.1: Search for conceptual code
Use mcp__project-watch-mcp__search_code with query="authentication and user login" and search_type="semantic"

# Expected output:
{
  "results": [
    {
      "file": "path/to/file.py",
      "line": <number>,
      "content": "...",
      "similarity": 0.85  # Higher = more relevant
    }
  ]
}

# Test 2.2: Search for test files
Use mcp__project-watch-mcp__search_code with query="unit tests and test cases" and search_type="semantic" and limit=5

# Test 2.3: Language-specific search
Use mcp__project-watch-mcp__search_code with query="class definitions" and language="python" and limit=10
```

### Test 3: Pattern Search

```bash
# Test 3.1: Find TODO comments
Use mcp__project-watch-mcp__search_code with query="TODO" and search_type="pattern"

# Test 3.2: Regex pattern search
Use mcp__project-watch-mcp__search_code with query="TODO|FIXME|HACK" and search_type="pattern" and is_regex=true

# Test 3.3: Find function definitions (Python)
Use mcp__project-watch-mcp__search_code with query="def .*\\(.*\\):" and search_type="pattern" and is_regex=true and language="python"
```

### Test 4: Repository Statistics

```bash
# Get comprehensive stats
Use mcp__project-watch-mcp__get_repository_stats

# Expected output structure:
{
  "total_files": <number>,
  "total_chunks": <number>,
  "total_size": <bytes>,
  "languages": {
    "python": {"files": <number>, "size": <bytes>, "percentage": <float>},
    "javascript": {...}
  },
  "largest_files": [
    {"path": "file.py", "size": <bytes>, "lines": <number>}
  ],
  "index_health": {
    "files_indexed": <number>,
    "index_coverage": <percentage>
  }
}
```

### Test 5: File Operations

```bash
# Test 5.1: Get file information
Use mcp__project-watch-mcp__get_file_info with file_path="src/project_watch_mcp/server.py"

# Expected output:
{
  "path": "src/project_watch_mcp/server.py",
  "language": "python",
  "size": <bytes>,
  "lines": <number>,
  "indexed": true,
  "chunk_count": <number>,
  "classes": [...],
  "functions": [...]
}

# Test 5.2: Refresh a file's index
Use mcp__project-watch-mcp__refresh_file with file_path="README.md"

# Expected output:
{
  "status": "success",
  "action": "updated",
  "chunks_after": <number>,
  "time_ms": <number>
}

# Test 5.3: Delete from index (NOT from disk)
Use mcp__project-watch-mcp__delete_file with file_path="tests/test_example.py"

# Expected output:
{
  "status": "success",
  "chunks_removed": <number>
}
```

### Test 6: Complexity Analysis

```bash
# Test 6.1: Analyze Python file complexity
Use mcp__project-watch-mcp__analyze_complexity with file_path="src/project_watch_mcp/server.py"

# Expected output:
{
  "file": "src/project_watch_mcp/server.py",
  "language": "python",
  "summary": {
    "total_complexity": <number>,
    "average_complexity": <float>,
    "maintainability_index": <float>,
    "complexity_grade": "A-F"
  },
  "functions": [
    {
      "name": "function_name",
      "complexity": <number>,
      "rank": "A-F",
      "line": <number>
    }
  ],
  "recommendations": [...]
}

# Test 6.2: Try on a non-Python file (should handle gracefully)
Use mcp__project-watch-mcp__analyze_complexity with file_path="README.md"

# Expected: Error or message that file type not supported
```

### Test 7: File Classification Features

```bash
# Test 7.1: List files by category
Use mcp__project-watch-mcp__list_files_by_category with category="test"

# Expected output:
[
  {
    "path": "tests/test_example.py",
    "language": "python",
    "size": 1234,
    "lines": 50,
    "category": "test",
    "is_test": true,
    "is_config": false,
    "is_documentation": false,
    "is_resource": false,
    "namespace": "tests.test_example"
  }
]

# Test 7.2: Search within specific file types
Use mcp__project-watch-mcp__search_code with query="function" and is_test=true

# Expected: Only results from test files

# Test 7.3: List configuration files
Use mcp__project-watch-mcp__list_files_by_category with category="config"

# Expected: Lists config.yaml, settings.json, .env files, etc.

# Test 7.4: Search excluding test files
Use mcp__project-watch-mcp__search_code with query="main" and is_test=false

# Expected: Results from source files only, no test files

# Test 7.5: List documentation files
Use mcp__project-watch-mcp__list_files_by_category with category="documentation"

# Expected: README.md, CHANGELOG.md, docs/*.md files
```

### Test 8: Monitoring Status

```bash
# Check monitoring and recent changes
Use mcp__project-watch-mcp__monitoring_status

# Expected output:
{
  "is_running": true,
  "repository_path": "/full/path/to/repo",
  "file_patterns": ["*.py", "*.js", ...],
  "monitoring_since": "2024-01-15T10:00:00Z",
  "pending_changes": <number>,
  "version_info": {
    "version": "0.1.0",
    "lucene_fix_version": "v2.0-double-escape"
  },
  "recent_changes": [
    {
      "change_type": "modified",
      "path": "file.py",
      "timestamp": "...",
      "processed": false
    }
  ]
}
```

## =� Performance Testing

### Large Repository Test

```bash
# Test on current repository
Use mcp__project-watch-mcp__initialize_repository

# Monitor indexing time for performance
Use mcp__project-watch-mcp__indexing_status
# Note the duration_seconds when complete

# Search performance test
Use mcp__project-watch-mcp__search_code with query="function implementation" and limit=50
# Response should be < 1 second

# Complex search test
Use mcp__project-watch-mcp__search_code with query="error handling try except finally" and search_type="semantic" and limit=100
# Even with 100 results, should be < 2 seconds
```

### Memory Usage Test

```bash
# Get baseline stats
Use mcp__project-watch-mcp__get_repository_stats
# Note total_chunks - this affects memory usage

# Perform multiple searches rapidly
Use mcp__project-watch-mcp__search_code with query="test1"
Use mcp__project-watch-mcp__search_code with query="test2"
Use mcp__project-watch-mcp__search_code with query="test3"
# Memory should remain stable, not increasing significantly
```

## =% Error Testing

### Test Invalid Inputs

```bash
# Test 1: Non-existent file
Use mcp__project-watch-mcp__get_file_info with file_path="does_not_exist.py"
# Expected: Error message about file not found

# Test 2: Invalid regex pattern
Use mcp__project-watch-mcp__search_code with query="[invalid(regex" and search_type="pattern" and is_regex=true
# Expected: Regex error message

# Test 3: File outside repository
Use mcp__project-watch-mcp__get_file_info with file_path="/etc/passwd"
# Expected: Error about file outside repository

# Test 4: Refresh non-existent file
Use mcp__project-watch-mcp__refresh_file with file_path="fake_file.py"
# Expected: File not found error

# Test 5: Delete already deleted file
Use mcp__project-watch-mcp__delete_file with file_path="already_deleted.py"
# Expected: Warning that file not in index
```

##  Verification Checklist

After running all tests, verify:

### Initialization
- [ ] Repository initializes without errors
- [ ] Indexing completes successfully
- [ ] File count matches expected repository size
- [ ] .gitignore patterns are respected

### Search Functionality
- [ ] Semantic search returns relevant results
- [ ] Pattern search finds exact matches
- [ ] Regex patterns work correctly
- [ ] Language filtering works
- [ ] Similarity scores make sense (higher for better matches)

### File Operations
- [ ] get_file_info returns accurate metadata
- [ ] refresh_file updates the index
- [ ] delete_file removes from index (not disk)
- [ ] File paths work with both relative and absolute

### Monitoring
- [ ] monitoring_status shows correct repository path
- [ ] Recent changes are tracked
- [ ] Version info is displayed
- [ ] Pending changes are processed

### Performance
- [ ] Search results return in < 1 second
- [ ] Large result sets (100 items) return in < 2 seconds
- [ ] Indexing completes at ~100 files/minute
- [ ] Memory usage remains stable

### Error Handling
- [ ] Invalid files return clear error messages
- [ ] Bad regex patterns are caught
- [ ] Files outside repository are rejected
- [ ] All errors include helpful messages

## =' Troubleshooting Test Failures

### Issue: "Unable to connect to Neo4j"

```bash
# Check if Neo4j is running
# On macOS: Check Neo4j Desktop app
# On Docker: docker ps | grep neo4j

# Test connection directly
curl -u neo4j:password http://localhost:7474
```

### Issue: "No files indexed"

```bash
# Check repository path
Use mcp__project-watch-mcp__monitoring_status
# Verify repository_path is correct

# Check file patterns
# Ensure your files match the monitored patterns
```

### Issue: "Search returns no results"

```bash
# First check if indexing is complete
Use mcp__project-watch-mcp__indexing_status

# Verify files are indexed
Use mcp__project-watch-mcp__get_repository_stats
# Check total_files > 0

# Try a simpler search
Use mcp__project-watch-mcp__search_code with query="def" and search_type="pattern"
```

### Issue: "Slow performance"

```bash
# Check pending changes
Use mcp__project-watch-mcp__monitoring_status
# High pending_changes indicates processing backlog

# Check total indexed files
Use mcp__project-watch-mcp__get_repository_stats
# Very large repositories (>10k files) may be slower
```

## =� Reporting Test Results

When reporting test results, include:

1. **Environment Info**:
   - Operating System
   - Neo4j version
   - Python version
   - Repository size (number of files)

2. **Test Results**:
   - Which tests passed/failed
   - Unexpected outputs
   - Performance metrics (timing)
   - Error messages received

3. **Specific Commands**:
   - Exact commands that failed
   - Full error output
   - Expected vs actual results

## <� Quick Test Suite

Run these commands in order for a complete test:

```bash
# 1. Initialize
Use mcp__project-watch-mcp__initialize_repository

# 2. Check status
Use mcp__project-watch-mcp__indexing_status

# 3. Get stats
Use mcp__project-watch-mcp__get_repository_stats

# 4. Semantic search
Use mcp__project-watch-mcp__search_code with query="main function entry point"

# 5. Pattern search
Use mcp__project-watch-mcp__search_code with query="TODO" and search_type="pattern"

# 6. File info
Use mcp__project-watch-mcp__get_file_info with file_path="README.md"

# 7. Complexity analysis (if Python files exist)
Use mcp__project-watch-mcp__analyze_complexity with file_path="src/project_watch_mcp/server.py"

# 8. Monitoring status
Use mcp__project-watch-mcp__monitoring_status
```

If all commands complete successfully, the MCP server is working correctly!