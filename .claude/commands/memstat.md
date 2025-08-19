# memstat - Project Watch MCP Status Command

## Overview
Monitor and interact with the Project Watch MCP repository monitoring tool.

## Available MCP Tools

### 1. Initialize Repository
```
mcp__project-watch-mcp__initialize_repository
```
Scans and indexes all repository files for semantic search.

### 2. Search Code
```
mcp__project-watch-mcp__search_code
```
**Parameters:**
- `query` (required): Search query
- `search_type`: "semantic" or "pattern" (default: "semantic")
- `is_regex`: For pattern search, treat as regex (default: false)
- `limit`: Max results (default: 10, max: 100)
- `language`: Filter by language (e.g., "python", "javascript")

### 3. Get Repository Stats
```
mcp__project-watch-mcp__get_repository_stats
```
Shows comprehensive repository statistics including file counts, languages, and index health.

### 4. Get File Info
```
mcp__project-watch-mcp__get_file_info
```
**Parameters:**
- `file_path` (required): Path to the file

### 5. Refresh File
```
mcp__project-watch-mcp__refresh_file
```
**Parameters:**
- `file_path` (required): Path to file to re-index

### 6. Monitoring Status
```
mcp__project-watch-mcp__monitoring_status
```
Shows real-time monitoring status and pending changes.

## Quick Status Commands

### Check Overall Status
```python
# Get current monitoring status
await mcp__project-watch-mcp__monitoring_status()

# Get repository statistics
await mcp__project-watch-mcp__get_repository_stats()
```

### Search Examples
```python
# Semantic search for authentication logic
await mcp__project-watch-mcp__search_code(
    query="user authentication and JWT validation",
    search_type="semantic",
    limit=5
)

# Pattern search for TODOs
await mcp__project-watch-mcp__search_code(
    query="TODO|FIXME|HACK",
    search_type="pattern",
    is_regex=True,
    limit=10
)

# Language-specific search
await mcp__project-watch-mcp__search_code(
    query="async function",
    language="typescript",
    limit=5
)
```

## Status Indicators

### 🟢 Healthy Status
- Repository initialized
- Files indexed successfully
- Monitoring active
- No pending changes
- Neo4j connected

### 🟡 Warning Status
- High pending changes count (>10)
- Some files failed to index
- Partial index coverage (<90%)

### 🔴 Error Status
- Neo4j disconnected
- Monitoring stopped
- Index initialization failed
- Repository path not found

## Quick Diagnostics

### 1. Check if MCP is running
```bash
# Check monitoring status
mcp__project-watch-mcp__monitoring_status()
```

### 2. View index statistics
```bash
# Get detailed stats
mcp__project-watch-mcp__get_repository_stats()
```

### 3. Verify file indexing
```bash
# Check specific file
mcp__project-watch-mcp__get_file_info("src/main.py")
```

### 4. Force re-index
```bash
# Re-initialize if needed
mcp__project-watch-mcp__initialize_repository()
```

## Common Issues & Solutions

### Issue: Files not being indexed
**Solution:** Check `.gitignore` patterns - ignored files won't be indexed

### Issue: Search returning no results
**Solution:** 
1. Verify repository is initialized
2. Check if files are properly indexed
3. Try different search types (semantic vs pattern)

### Issue: Neo4j connection failed
**Solution:** 
1. Ensure Neo4j Desktop is running
2. Verify environment variables are set
3. Check Neo4j credentials

### Issue: High memory usage
**Solution:**
1. Large repositories may need Neo4j tuning
2. Consider limiting file patterns monitored
3. Check chunk size configuration

## Environment Requirements

**Required Environment Variables:**
```bash
NEO4J_URI=neo4j://127.0.0.1:7687
PROJECT_WATCH_USER=neo4j
PROJECT_WATCH_PASSWORD=<your-password>
NEO4J_DB=memory
```

**File Patterns Monitored:**
- `*.py`, `*.js`, `*.ts`, `*.jsx`, `*.tsx`
- `*.java`, `*.cpp`, `*.c`, `*.h`, `*.hpp`
- `*.cs`, `*.go`, `*.rs`, `*.rb`, `*.php`
- `*.swift`, `*.kt`, `*.scala`, `*.r`, `*.m`
- `*.sql`, `*.sh`, `*.yaml`, `*.yml`
- `*.toml`, `*.json`, `*.xml`, `*.html`
- `*.css`, `*.scss`, `*.md`, `*.txt`

## Usage Tips

1. **Initialize first**: Always run `initialize_repository` when starting
2. **Semantic vs Pattern**: Use semantic for concepts, pattern for exact matches
3. **Monitor changes**: Check `monitoring_status` for pending updates
4. **Refresh on issues**: Use `refresh_file` if a file seems out of sync
5. **Check stats regularly**: `get_repository_stats` shows index health

## Example Workflow

```python
# 1. Initialize repository
await mcp__project-watch-mcp__initialize_repository()

# 2. Check status
status = await mcp__project-watch-mcp__monitoring_status()
print(f"Monitoring: {status['is_running']}")
print(f"Files monitored: {status['statistics']['files_monitored']}")

# 3. Search for specific code
results = await mcp__project-watch-mcp__search_code(
    query="database connection handling",
    search_type="semantic",
    limit=5
)

# 4. Get file details
for result in results['results']:
    info = await mcp__project-watch-mcp__get_file_info(result['file'])
    print(f"{info['path']}: {info['lines']} lines, {info['chunk_count']} chunks")

# 5. Check overall health
stats = await mcp__project-watch-mcp__get_repository_stats()
print(f"Total indexed: {stats['total_files']} files, {stats['total_chunks']} chunks")
```