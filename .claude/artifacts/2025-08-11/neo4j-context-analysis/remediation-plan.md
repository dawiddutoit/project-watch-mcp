# REMEDIATION PLAN: Adding Project Context to Neo4j RAG System

## Overview
This document provides a step-by-step remediation plan to fix the critical project context isolation issues identified in the Neo4j RAG system.

---

## PHASE 1: IMMEDIATE HOTFIX (Day 1)
*Prevent further data corruption*

### Step 1.1: Add Warning to Documentation

**File: README.md**
```markdown
## ⚠️ CRITICAL LIMITATION ⚠️

**This system currently supports ONLY ONE PROJECT per Neo4j instance.**

Multiple projects sharing the same Neo4j database WILL experience:
- Data corruption and overwriting
- Mixed search results across projects  
- Security breaches from code leakage

**DO NOT USE** in multi-project environments until version 0.2.0.

For multi-project support, you must:
1. Use separate Neo4j instances per project, OR
2. Wait for version 0.2.0 with project isolation support
```

### Step 1.2: Add Runtime Warning

**File: src/project_watch_mcp/cli.py**
```python
async def main(...):
    """Main entry point for the Project Watch MCP server."""
    
    # Add warning at startup
    logger.warning("=" * 60)
    logger.warning("CRITICAL: This system supports ONLY ONE PROJECT per Neo4j instance")
    logger.warning("Multiple projects will cause DATA CORRUPTION")
    logger.warning("See documentation for multi-project deployment options")
    logger.warning("=" * 60)
    
    # Rest of existing code...
```

---

## PHASE 2: DATA MODEL UPDATES (Day 2-3)

### Step 2.1: Update Data Classes

**File: src/project_watch_mcp/neo4j_rag.py**

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProjectContext:
    """Represents project context for isolation."""
    name: str
    repository_path: Path
    created_at: datetime
    last_indexed: Optional[datetime] = None
    
    @property
    def id(self) -> str:
        """Generate unique project ID from name and path."""
        return hashlib.sha256(
            f"{self.name}:{self.repository_path}".encode()
        ).hexdigest()[:16]

@dataclass
class CodeFile:
    """Represents a code file to be indexed."""
    path: Path
    content: str
    language: str
    size: int
    last_modified: datetime
    project_context: ProjectContext  # NEW: Required field
    
    @property
    def file_hash(self) -> str:
        """Generate a hash of the file content."""
        return hashlib.sha256(self.content.encode()).hexdigest()
    
    @property
    def unique_id(self) -> str:
        """Generate unique ID including project context."""
        return f"{self.project_context.id}:{self.path}"

@dataclass
class CodeChunk:
    """Represents a chunk of code."""
    file_path: Path
    content: str
    start_line: int
    end_line: int
    project_context: ProjectContext  # NEW: Required field
    embedding: list[float] | None = None
    
    @property
    def unique_id(self) -> str:
        """Generate unique ID including project context."""
        return f"{self.project_context.id}:{self.file_path}:{self.start_line}"
```

### Step 2.2: Update Neo4jRAG Class

**File: src/project_watch_mcp/neo4j_rag.py**

```python
class Neo4jRAG:
    """Neo4j-based RAG system for code retrieval with project isolation."""
    
    def __init__(
        self,
        neo4j_driver: AsyncDriver,
        project_context: ProjectContext,  # NEW: Required parameter
        embeddings: EmbeddingsProvider | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """Initialize the Neo4j RAG system with project context."""
        self.neo4j_driver = neo4j_driver
        self.project_context = project_context  # NEW: Store context
        self.embeddings = embeddings or EmbeddingsProvider()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    async def initialize(self):
        """Initialize the RAG system and create necessary indexes."""
        await self.create_project_node()  # NEW: Create project node
        await self.create_indexes()
    
    async def create_project_node(self):
        """Create or update the project node."""
        query = """
        MERGE (p:Project {id: $project_id})
        SET p.name = $name,
            p.repository_path = $repo_path,
            p.created_at = COALESCE(p.created_at, $created_at),
            p.last_indexed = $last_indexed
        RETURN p
        """
        
        await self.neo4j_driver.execute_query(
            query,
            {
                "project_id": self.project_context.id,
                "name": self.project_context.name,
                "repo_path": str(self.project_context.repository_path),
                "created_at": self.project_context.created_at.isoformat(),
                "last_indexed": datetime.now().isoformat(),
            },
            routing_control=RoutingControl.WRITE,
        )
    
    async def create_indexes(self):
        """Create Neo4j indexes for efficient querying with project isolation."""
        queries = [
            # Composite index for project + file path
            """
            CREATE INDEX file_project_path_index IF NOT EXISTS
            FOR (f:CodeFile) ON (f.project_id, f.path)
            """,
            # Ensure unique files per project
            """
            CREATE CONSTRAINT unique_file_per_project IF NOT EXISTS
            FOR (f:CodeFile) REQUIRE (f.project_id, f.path) IS UNIQUE
            """,
            # Project index
            """
            CREATE INDEX project_id_index IF NOT EXISTS
            FOR (p:Project) ON (p.id)
            """,
            # Language index scoped by project
            """
            CREATE INDEX file_project_language_index IF NOT EXISTS
            FOR (f:CodeFile) ON (f.project_id, f.language)
            """,
            # Fulltext index (Note: needs special handling for project scoping)
            """
            CREATE FULLTEXT INDEX code_search IF NOT EXISTS
            FOR (c:CodeChunk) ON EACH [c.content, c.project_id]
            """,
        ]
        
        for query in queries:
            try:
                await self.neo4j_driver.execute_query(
                    query, routing_control=RoutingControl.WRITE
                )
                logger.debug(f"Created index: {query.split()[2]}")
            except Exception as e:
                logger.debug(f"Index creation: {e}")
```

### Step 2.3: Update Index File Method

```python
async def index_file(self, code_file: CodeFile):
    """Index a code file in Neo4j with project isolation."""
    
    # Ensure project context matches
    if code_file.project_context.id != self.project_context.id:
        raise ValueError(
            f"File project context {code_file.project_context.id} "
            f"doesn't match RAG context {self.project_context.id}"
        )
    
    # Create or update file node WITH PROJECT CONTEXT
    file_query = """
    MATCH (p:Project {id: $project_id})
    MERGE (f:CodeFile {path: $path, project_id: $project_id})
    SET f.language = $language,
        f.size = $size,
        f.last_modified = $last_modified,
        f.hash = $hash,
        f.project_name = $project_name
    MERGE (p)-[:HAS_FILE]->(f)
    RETURN f
    """
    
    await self.neo4j_driver.execute_query(
        file_query,
        {
            "project_id": self.project_context.id,
            "project_name": self.project_context.name,
            "path": str(code_file.path),
            "language": code_file.language,
            "size": code_file.size,
            "last_modified": code_file.last_modified.isoformat(),
            "hash": code_file.file_hash,
        },
        routing_control=RoutingControl.WRITE,
    )
    
    # Delete existing chunks for this file IN THIS PROJECT
    delete_chunks_query = """
    MATCH (f:CodeFile {path: $path, project_id: $project_id})-[:HAS_CHUNK]->(c:CodeChunk)
    DETACH DELETE c
    """
    
    await self.neo4j_driver.execute_query(
        delete_chunks_query,
        {
            "path": str(code_file.path),
            "project_id": self.project_context.id,
        },
        routing_control=RoutingControl.WRITE,
    )
    
    # Create chunks with project context
    chunks = self.chunk_content(code_file.content, self.chunk_size, self.chunk_overlap)
    
    for i, chunk in enumerate(chunks):
        # ... chunk processing ...
        
        chunk_query = """
        MATCH (f:CodeFile {path: $file_path, project_id: $project_id})
        CREATE (c:CodeChunk {
            content: $content,
            start_line: $start_line,
            end_line: $end_line,
            embedding: $embedding,
            chunk_index: $chunk_index,
            project_id: $project_id,
            project_name: $project_name
        })
        CREATE (f)-[:HAS_CHUNK]->(c)
        """
        
        await self.neo4j_driver.execute_query(
            chunk_query,
            {
                "file_path": str(code_file.path),
                "project_id": self.project_context.id,
                "project_name": self.project_context.name,
                "content": chunk,
                "start_line": start_line,
                "end_line": end_line,
                "embedding": embedding,
                "chunk_index": i,
            },
            routing_control=RoutingControl.WRITE,
        )
```

### Step 2.4: Update Search Methods

```python
async def search_semantic(
    self,
    query: str,
    limit: int = 10,
    language: str | None = None,
) -> list[SearchResult]:
    """Perform semantic search with project isolation."""
    
    query_embedding = await self.embeddings.embed_text(query)
    
    # Vector search WITH PROJECT FILTER
    cypher = """
    CALL db.index.vector.queryNodes('chunk_embeddings', $k, $query_embedding)
    YIELD node as c, score
    MATCH (p:Project {id: $project_id})-[:HAS_FILE]->(f:CodeFile)-[:HAS_CHUNK]->(c)
    """
    
    if language:
        cypher += " WHERE f.language = $language"
    
    cypher += """
    RETURN f.path as file_path,
           c.content as chunk_content,
           c.start_line as line_number,
           score as similarity
    ORDER BY score DESC
    """
    
    params = {
        "query_embedding": query_embedding,
        "k": limit * 3,  # Get more results to filter by project
        "project_id": self.project_context.id,
    }
    
    if language:
        params["language"] = language
    
    result = await self.neo4j_driver.execute_query(
        cypher, params, routing_control=RoutingControl.READ
    )
    
    # ... rest of method
```

---

## PHASE 3: SERVER UPDATES (Day 3-4)

### Step 3.1: Update Server Creation

**File: src/project_watch_mcp/server.py**

```python
def create_mcp_server(
    repository_monitor: RepositoryMonitor,
    neo4j_rag: Neo4jRAG,
    project_context: ProjectContext,  # NEW: Add project context
) -> FastMCP:
    """Create an MCP server with project isolation."""
    
    server_name = f"project-watch-mcp-{project_context.name}"
    mcp = FastMCP(server_name, dependencies=["neo4j", "watchfiles", "pydantic"])
    
    # Update all tool implementations to use project context...
```

### Step 3.2: Update CLI

**File: src/project_watch_mcp/cli.py**

```python
async def main(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str,
    repository_path: str,
    project_name: str = None,  # NEW: Optional project name
    transport: Literal["stdio", "sse", "http"] = "stdio",
    # ... other params
) -> None:
    """Main entry point with project isolation support."""
    
    # Derive project name if not provided
    if not project_name:
        repo_path = Path(repository_path).resolve()
        project_name = repo_path.name
    
    # Create project context
    project_context = ProjectContext(
        name=project_name,
        repository_path=Path(repository_path),
        created_at=datetime.now(),
    )
    
    logger.info(f"Project: {project_context.name} (ID: {project_context.id})")
    
    # Create Neo4j RAG with project context
    neo4j_rag = Neo4jRAG(
        neo4j_driver=neo4j_driver,
        project_context=project_context,  # Pass project context
        chunk_size=100,
        chunk_overlap=20,
    )
    
    # ... rest of initialization
```

---

## PHASE 4: TESTING (Day 4-5)

### Step 4.1: Add Integration Tests

**File: tests/test_project_isolation.py**

```python
import pytest
from pathlib import Path
from datetime import datetime
from src.project_watch_mcp.neo4j_rag import Neo4jRAG, CodeFile, ProjectContext

@pytest.mark.asyncio
async def test_project_isolation(neo4j_driver):
    """Test that projects are properly isolated."""
    
    # Create two project contexts
    project_a = ProjectContext(
        name="project_a",
        repository_path=Path("/projects/a"),
        created_at=datetime.now()
    )
    
    project_b = ProjectContext(
        name="project_b", 
        repository_path=Path("/projects/b"),
        created_at=datetime.now()
    )
    
    # Create RAG instances for each project
    rag_a = Neo4jRAG(neo4j_driver, project_a)
    rag_b = Neo4jRAG(neo4j_driver, project_b)
    
    await rag_a.initialize()
    await rag_b.initialize()
    
    # Index same file path in both projects
    file_a = CodeFile(
        path=Path("src/main.py"),
        content="Project A main",
        language="python",
        size=100,
        last_modified=datetime.now(),
        project_context=project_a
    )
    
    file_b = CodeFile(
        path=Path("src/main.py"),
        content="Project B main",
        language="python",
        size=200,
        last_modified=datetime.now(),
        project_context=project_b
    )
    
    await rag_a.index_file(file_a)
    await rag_b.index_file(file_b)
    
    # Search in project A should only return A's results
    results_a = await rag_a.search_semantic("main")
    assert all("Project A" in r.content for r in results_a)
    assert not any("Project B" in r.content for r in results_a)
    
    # Search in project B should only return B's results
    results_b = await rag_b.search_semantic("main")
    assert all("Project B" in r.content for r in results_b)
    assert not any("Project A" in r.content for r in results_b)

@pytest.mark.asyncio
async def test_no_cross_project_contamination(neo4j_driver):
    """Test that operations in one project don't affect another."""
    # ... test implementation
```

---

## PHASE 5: MIGRATION (Day 5-6)

### Step 5.1: Migration Script

**File: scripts/migrate_to_project_context.py**

```python
#!/usr/bin/env python3
"""
Migration script to add project context to existing Neo4j data.
"""

import asyncio
from neo4j import AsyncGraphDatabase
import click

@click.command()
@click.option('--neo4j-uri', default='bolt://localhost:7687')
@click.option('--neo4j-user', default='neo4j')
@click.option('--neo4j-password', required=True)
@click.option('--project-name', required=True, help='Name for existing data')
@click.option('--dry-run', is_flag=True, help='Preview changes without applying')
async def migrate(neo4j_uri, neo4j_user, neo4j_password, project_name, dry_run):
    """Migrate existing Neo4j data to include project context."""
    
    driver = AsyncGraphDatabase.driver(
        neo4j_uri, auth=(neo4j_user, neo4j_password)
    )
    
    try:
        await driver.verify_connectivity()
        print(f"✓ Connected to Neo4j at {neo4j_uri}")
        
        # Count existing nodes
        count_result = await driver.execute_query("""
            MATCH (f:CodeFile)
            WHERE NOT EXISTS(f.project_id)
            RETURN COUNT(f) as file_count
        """)
        
        file_count = count_result.records[0]['file_count']
        print(f"Found {file_count} files without project context")
        
        if dry_run:
            print("DRY RUN - No changes will be made")
            return
        
        if not click.confirm(f"Add project '{project_name}' to {file_count} files?"):
            print("Migration cancelled")
            return
        
        # Create project node
        project_id = hashlib.sha256(f"{project_name}:legacy".encode()).hexdigest()[:16]
        
        await driver.execute_query("""
            MERGE (p:Project {id: $project_id})
            SET p.name = $project_name,
                p.repository_path = 'legacy_migration',
                p.created_at = datetime(),
                p.migrated = true
        """, project_id=project_id, project_name=project_name)
        
        # Update CodeFile nodes
        await driver.execute_query("""
            MATCH (f:CodeFile)
            WHERE NOT EXISTS(f.project_id)
            SET f.project_id = $project_id,
                f.project_name = $project_name
        """, project_id=project_id, project_name=project_name)
        
        # Update CodeChunk nodes
        await driver.execute_query("""
            MATCH (c:CodeChunk)
            WHERE NOT EXISTS(c.project_id)
            SET c.project_id = $project_id,
                c.project_name = $project_name
        """, project_id=project_id, project_name=project_name)
        
        # Create relationships
        await driver.execute_query("""
            MATCH (p:Project {id: $project_id})
            MATCH (f:CodeFile {project_id: $project_id})
            MERGE (p)-[:HAS_FILE]->(f)
        """, project_id=project_id)
        
        print(f"✓ Migration complete for project '{project_name}'")
        
    finally:
        await driver.close()

if __name__ == "__main__":
    asyncio.run(migrate())
```

---

## PHASE 6: DOCUMENTATION (Day 6-7)

### Step 6.1: Update README

```markdown
## Project Isolation (v0.2.0+)

Project Watch MCP now supports multiple projects in the same Neo4j instance through project context isolation.

### How It Works

Each indexed repository is assigned a unique project context that:
- Isolates all data (files, chunks, embeddings) by project
- Prevents cross-project data contamination
- Enables project-specific searches and statistics
- Maintains data integrity across multiple projects

### Configuration

```bash
# Specify project name explicitly
project-watch-mcp --repository /path/to/repo --project-name "my-project"

# Or let it auto-derive from repository name
project-watch-mcp --repository /path/to/my-project
```

### Migration from v0.1.x

If upgrading from v0.1.x with existing data:

```bash
# Run migration script
python scripts/migrate_to_project_context.py \
  --neo4j-password your_password \
  --project-name "legacy_project"
```
```

---

## VALIDATION CHECKLIST

- [ ] All nodes include project_id field
- [ ] All queries filter by project_id
- [ ] Composite indexes created for (project_id, path)
- [ ] Unique constraints enforce project-level uniqueness
- [ ] Search operations are project-scoped
- [ ] Statistics are project-specific
- [ ] Migration script handles existing data
- [ ] Tests verify project isolation
- [ ] Documentation updated with warnings
- [ ] Performance impact assessed

---

## ROLLBACK PLAN

If issues arise after deployment:

1. **Immediate**: Revert to single-project warning mode
2. **Data Recovery**: Use backup to restore pre-migration state
3. **Gradual Migration**: Move projects one at a time
4. **Monitoring**: Track query performance and data integrity

---

## SUCCESS METRICS

- Zero cross-project data contamination incidents
- Search results 100% project-accurate
- No performance degradation >10%
- All integration tests passing
- Zero data loss during migration

---

*This remediation plan provides a complete path from the current broken state to a properly isolated multi-project system.*