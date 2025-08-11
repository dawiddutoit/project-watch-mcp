# CODE LOCATIONS SUMMARY: Project Context Issues

## Critical Code Locations Where Context is Missing

### 1. **neo4j_rag.py** - Core Data Structures

#### Lines 18-31: CodeFile class
```python
@dataclass
class CodeFile:
    path: Path  # ❌ No project identifier
    content: str
    language: str
    size: int
    last_modified: datetime
    # MISSING: project_name or project_id field
```

#### Lines 34-43: CodeChunk class  
```python
@dataclass
class CodeChunk:
    file_path: Path  # ❌ No project context
    content: str
    start_line: int
    end_line: int
    embedding: list[float] | None = None
    # MISSING: project_name or project_id field
```

### 2. **neo4j_rag.py** - Data Creation Methods

#### Lines 176-195: File node creation
```python
file_query = """
MERGE (f:CodeFile {path: $path})  # ❌ Only path as unique key
SET f.language = $language,
    f.size = $size,
    f.last_modified = $last_modified,
    f.hash = $hash
RETURN f
"""
# MISSING: project_name in MERGE and SET clauses
```

#### Lines 222-245: Chunk node creation
```python
chunk_query = """
MATCH (f:CodeFile {path: $file_path})  # ❌ No project filter
CREATE (c:CodeChunk {
    content: $content,
    start_line: $start_line,
    end_line: $end_line,
    embedding: $embedding,
    chunk_index: $chunk_index
    # MISSING: project_id or project_name
})
CREATE (f)-[:HAS_CHUNK]->(c)
"""
```

### 3. **neo4j_rag.py** - Search Methods

#### Lines 291-395: search_semantic method
```python
async def search_semantic(
    self,
    query: str,
    limit: int = 10,
    language: str | None = None,  # ❌ No project_name parameter
) -> list[SearchResult]:
    # ...
    cypher = """
    CALL db.index.vector.queryNodes('chunk_embeddings', $k, $query_embedding)
    YIELD node as c, score
    MATCH (f:CodeFile)-[:HAS_CHUNK]->(c)  # ❌ No project filter
    """
```

#### Lines 397-453: search_by_pattern method
```python
async def search_by_pattern(
    self,
    pattern: str,
    is_regex: bool = False,
    limit: int = 10,  # ❌ No project_name parameter
) -> list[SearchResult]:
    # ...
    query = """
    MATCH (f:CodeFile)-[:HAS_CHUNK]->(c:CodeChunk)  # ❌ No project filter
    WHERE c.content =~ $pattern
    """
```

### 4. **neo4j_rag.py** - File Operations

#### Lines 251-270: update_file method
```python
async def update_file(self, code_file: CodeFile):  # ❌ No project context
    check_query = """
    MATCH (f:CodeFile {path: $path})  # ❌ Will match across all projects
    RETURN f.hash as hash
    """
```

#### Lines 272-289: delete_file method
```python
async def delete_file(self, file_path: Path):  # ❌ No project parameter
    query = """
    MATCH (f:CodeFile {path: $path})  # ❌ Will delete from all projects!
    OPTIONAL MATCH (f)-[:HAS_CHUNK]->(c:CodeChunk)
    DETACH DELETE f, c
    """
```

### 5. **neo4j_rag.py** - Statistics

#### Lines 493-525: get_repository_stats method
```python
async def get_repository_stats(self) -> dict[str, Any]:
    query = """
    MATCH (f:CodeFile)  # ❌ Counts ALL files across ALL projects
    OPTIONAL MATCH (f)-[:HAS_CHUNK]->(c:CodeChunk)
    RETURN count(DISTINCT f) as total_files,
           count(c) as total_chunks
    """
```

### 6. **server.py** - MCP Tool Implementations

#### Lines 47-91: initialize_repository tool
```python
async def initialize_repository() -> str:
    # ...
    code_file = CodeFile(
        path=file_info.path,
        content=content,
        language=file_info.language,
        size=file_info.size,
        last_modified=file_info.last_modified,
        # ❌ No project_context field
    )
    await neo4j_rag.index_file(code_file)  # ❌ No project context passed
```

#### Lines 102-160: search_code tool
```python
async def search_code(
    query: str = Field(..., description="Search query"),
    search_type: Literal["semantic", "pattern"] = Field(...),
    is_regex: bool = Field(...),
    limit: int = Field(...),
    language: str | None = Field(...),  # ❌ No project_name parameter
) -> list[dict]:
```

### 7. **repository_monitor.py** - File Tracking

#### Lines 27-76: FileInfo class
```python
@dataclass
class FileInfo:
    path: Path
    size: int
    last_modified: datetime
    language: str | None = None
    # ❌ No project_name or project_id field
```

#### Lines 79-90: FileChangeEvent class
```python
@dataclass
class FileChangeEvent:
    path: Path
    change_type: FileChangeType
    timestamp: datetime = None
    # ❌ No project context
```

### 8. **cli.py** - System Initialization

#### Lines 23-49: main function signature
```python
async def main(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str,
    repository_path: str,  # ❌ No project_name parameter
    transport: Literal["stdio", "sse", "http"] = "stdio",
    # ...
) -> None:
```

#### Lines 73-92: Component initialization
```python
# Create repository monitor
repository_monitor = RepositoryMonitor(
    repo_path=Path(repository_path),
    neo4j_driver=neo4j_driver,
    file_patterns=patterns,
    # ❌ No project context
)

# Create Neo4j RAG system
neo4j_rag = Neo4jRAG(
    neo4j_driver=neo4j_driver,
    chunk_size=100,
    chunk_overlap=20,
    # ❌ No project context
)

# Create MCP server
mcp = create_mcp_server(
    repository_monitor=repository_monitor,
    neo4j_rag=neo4j_rag,
    # ❌ No project context
)
```

### 9. **Neo4j Indexes** - Schema Issues

#### Lines 98-140: Index creation in neo4j_rag.py
```python
# File path index - not unique per project
CREATE INDEX file_path_index IF NOT EXISTS
FOR (f:CodeFile) ON (f.path)

# Vector index - no project boundaries
CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
FOR (c:CodeChunk) ON (c.embedding)

# Fulltext index - searches across all projects
CREATE FULLTEXT INDEX code_search IF NOT EXISTS
FOR (c:CodeChunk) ON EACH [c.content]

# ❌ MISSING: Composite indexes for (project_id, path)
# ❌ MISSING: Unique constraints per project
# ❌ MISSING: Project-scoped vector index
```

## Summary of Missing Context Locations

| File | Method/Class | Lines | Issue |
|------|-------------|-------|-------|
| neo4j_rag.py | CodeFile | 18-31 | No project_id field |
| neo4j_rag.py | CodeChunk | 34-43 | No project_id field |
| neo4j_rag.py | index_file | 176-245 | No project in node creation |
| neo4j_rag.py | update_file | 251-270 | No project filtering |
| neo4j_rag.py | delete_file | 272-289 | Deletes across all projects |
| neo4j_rag.py | search_semantic | 291-395 | No project filtering |
| neo4j_rag.py | search_by_pattern | 397-453 | No project filtering |
| neo4j_rag.py | get_repository_stats | 493-525 | Returns global stats |
| neo4j_rag.py | create_indexes | 98-140 | No project-scoped indexes |
| server.py | initialize_repository | 47-91 | No project context passed |
| server.py | search_code | 102-160 | No project parameter |
| server.py | create_mcp_server | 20-351 | No project context parameter |
| repository_monitor.py | FileInfo | 27-76 | No project field |
| repository_monitor.py | FileChangeEvent | 79-90 | No project field |
| cli.py | main | 23-116 | No project_name parameter |

## Critical Query Patterns That Need Fixing

### Current (BROKEN):
```cypher
MATCH (f:CodeFile {path: $path})
```

### Fixed:
```cypher
MATCH (f:CodeFile {path: $path, project_id: $project_id})
```

### Current (BROKEN):
```cypher
MATCH (f:CodeFile)-[:HAS_CHUNK]->(c:CodeChunk)
```

### Fixed:
```cypher
MATCH (p:Project {id: $project_id})-[:HAS_FILE]->(f:CodeFile)-[:HAS_CHUNK]->(c:CodeChunk)
```

## Impact Assessment by Component

1. **Data Layer (neo4j_rag.py)**: 100% affected - every method needs updating
2. **Server Layer (server.py)**: 100% affected - all tools need project context
3. **Monitor Layer (repository_monitor.py)**: Structural changes needed
4. **CLI Layer (cli.py)**: Parameter additions required
5. **Test Layer**: New tests needed for project isolation

---

*Every single data operation in the system currently lacks project context, making multi-project deployment impossible without data corruption.*