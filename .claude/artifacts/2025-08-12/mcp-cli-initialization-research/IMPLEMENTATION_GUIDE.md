# Implementation Guide: CLI Initialization for Project Watch MCP

## Complete Refactoring Plan

### Phase 1: Extract Core Logic (2 hours)

#### 1.1 Create Core Initializer Module
```python
# src/project_watch_mcp/core/__init__.py
"""Core business logic independent of transport layer."""

# src/project_watch_mcp/core/initializer.py
"""Repository initialization logic extracted from server."""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass

from ..neo4j_rag import Neo4jRAG, CodeFile
from ..repository_monitor import RepositoryMonitor, FileInfo

logger = logging.getLogger(__name__)


@dataclass
class InitializationResult:
    """Result of repository initialization."""
    indexed: int
    total: int
    failed: List[Path]
    message: str
    monitoring_started: bool


class RepositoryInitializer:
    """
    Handles repository initialization independent of MCP server context.
    
    This class encapsulates all initialization logic, making it reusable
    from CLI, MCP tools, and hooks without code duplication.
    """
    
    def __init__(
        self,
        repository_monitor: RepositoryMonitor,
        neo4j_rag: Neo4jRAG,
        project_name: str
    ):
        """
        Initialize with dependencies.
        
        Args:
            repository_monitor: Monitor for file system operations
            neo4j_rag: RAG system for semantic indexing
            project_name: Project identifier for context isolation
        """
        self.monitor = repository_monitor
        self.rag = neo4j_rag
        self.project_name = project_name
        self._is_initialized = False
    
    async def initialize(
        self,
        start_monitoring: bool = True,
        progress_callback: Optional[callable] = None
    ) -> InitializationResult:
        """
        Initialize repository indexing and optionally start monitoring.
        
        Args:
            start_monitoring: Whether to start file monitoring after indexing
            progress_callback: Optional callback for progress updates
                              Signature: (current: int, total: int, file: Path)
        
        Returns:
            InitializationResult with statistics and status
        """
        try:
            # Scan repository for matching files
            logger.info(f"Scanning repository for project: {self.project_name}")
            files = await self.monitor.scan_repository()
            total_files = len(files)
            
            # Track results
            indexed_count = 0
            failed_files = []
            
            # Index each file
            for idx, file_info in enumerate(files, 1):
                try:
                    # Report progress if callback provided
                    if progress_callback:
                        await progress_callback(idx, total_files, file_info.path)
                    
                    # Read file content
                    content = file_info.path.read_text(encoding="utf-8")
                    
                    # Create CodeFile object
                    code_file = CodeFile(
                        project_name=self.project_name,
                        path=file_info.path,
                        content=content,
                        language=file_info.language,
                        size=file_info.size,
                        last_modified=file_info.last_modified,
                    )
                    
                    # Index in Neo4j
                    await self.rag.index_file(code_file)
                    indexed_count += 1
                    logger.debug(f"Indexed: {file_info.path}")
                    
                except UnicodeDecodeError:
                    logger.warning(f"Skipping binary file: {file_info.path}")
                    failed_files.append(file_info.path)
                except Exception as e:
                    logger.error(f"Failed to index {file_info.path}: {e}")
                    failed_files.append(file_info.path)
            
            # Start monitoring if requested
            monitoring_started = False
            if start_monitoring:
                try:
                    await self.monitor.start()
                    monitoring_started = True
                    logger.info("File monitoring started")
                except Exception as e:
                    logger.error(f"Failed to start monitoring: {e}")
            
            # Mark as initialized
            self._is_initialized = True
            
            # Prepare result message
            if indexed_count == total_files:
                message = f"Successfully indexed all {total_files} files"
            else:
                message = f"Indexed {indexed_count}/{total_files} files ({len(failed_files)} failed)"
            
            return InitializationResult(
                indexed=indexed_count,
                total=total_files,
                failed=failed_files,
                message=message,
                monitoring_started=monitoring_started
            )
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    async def refresh_file(self, file_path: Path) -> bool:
        """
        Refresh a single file in the index.
        
        Args:
            file_path: Path to file to refresh
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Implementation here
            pass
        except Exception as e:
            logger.error(f"Failed to refresh {file_path}: {e}")
            return False
    
    @property
    def is_initialized(self) -> bool:
        """Check if repository has been initialized."""
        return self._is_initialized
```

#### 1.2 Update CLI with Initialization Support
```python
# src/project_watch_mcp/cli.py (modified sections only)

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Literal

from neo4j import AsyncGraphDatabase

from .config import EmbeddingConfig, ProjectConfig
from .core.initializer import RepositoryInitializer  # NEW
from .neo4j_rag import Neo4jRAG
from .repository_monitor import RepositoryMonitor
from .server import create_mcp_server
from .utils.embedding import create_embeddings_provider

# ... existing imports and setup ...

async def initialize_only(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str,
    repository_path: str,
    project_name: str | None = None,
    file_patterns: str = "*.py,*.js,*.ts,*.java,*.go,*.rs,*.md,*.json,*.yaml,*.yml,*.toml",
    verbose: bool = False
) -> int:
    """
    Initialize repository without starting server.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Setup progress reporting for CLI
    async def progress_reporter(current: int, total: int, file: Path):
        if verbose:
            print(f"[{current}/{total}] Indexing: {file.relative_to(repository_path)}")
        elif current % 10 == 0 or current == total:
            print(f"Progress: {current}/{total} files indexed", end='\r')
    
    try:
        # Connect to Neo4j
        neo4j_driver = AsyncGraphDatabase.driver(
            neo4j_uri, auth=(neo4j_user, neo4j_password), database=neo4j_database
        )
        
        # Verify connection
        await neo4j_driver.verify_connectivity()
        
        # Create project configuration
        if project_name:
            project_config = ProjectConfig(name=project_name, repository_path=Path(repository_path))
        else:
            project_config = ProjectConfig.from_repository_path(Path(repository_path))
        
        # Parse file patterns
        patterns = [p.strip() for p in file_patterns.split(",")]
        
        # Create repository monitor
        repository_monitor = RepositoryMonitor(
            repo_path=Path(repository_path),
            project_name=project_config.name,
            neo4j_driver=neo4j_driver,
            file_patterns=patterns,
        )
        
        # Create embeddings
        embedding_config = EmbeddingConfig.from_env()
        embeddings = create_embeddings_provider(
            provider_type=embedding_config.provider,
            api_key=embedding_config.openai_api_key,
            model=embedding_config.openai_model,
            api_url=embedding_config.local_api_url,
            dimension=embedding_config.dimension,
        )
        
        # Create Neo4j RAG system
        neo4j_rag = Neo4jRAG(
            neo4j_driver=neo4j_driver,
            project_name=project_config.name,
            embeddings=embeddings,
            chunk_size=100,
            chunk_overlap=20,
        )
        await neo4j_rag.initialize()
        
        # Create and run initializer
        initializer = RepositoryInitializer(
            repository_monitor=repository_monitor,
            neo4j_rag=neo4j_rag,
            project_name=project_config.name
        )
        
        print(f"Initializing repository: {repository_path}")
        print(f"Project name: {project_config.name}")
        
        result = await initializer.initialize(
            start_monitoring=False,  # Don't start monitoring in CLI mode
            progress_callback=progress_reporter if verbose else None
        )
        
        # Clear progress line and show final result
        print("\033[K", end='')  # Clear line
        print(result.message)
        
        if result.failed:
            print(f"Failed files: {len(result.failed)}")
            if verbose:
                for path in result.failed[:10]:  # Show first 10
                    print(f"  - {path.relative_to(repository_path)}")
                if len(result.failed) > 10:
                    print(f"  ... and {len(result.failed) - 10} more")
        
        # Cleanup
        await neo4j_driver.close()
        
        return 0 if result.indexed > 0 else 1
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cli():
    """Command-line interface for Project Watch."""
    parser = argparse.ArgumentParser(
        prog="project-watch-mcp",
        description="Project Watch - Repository Monitoring MCP Server with Neo4j RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""... existing epilog ..."""
    )
    
    # Add initialization flag (mutually exclusive with transport)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--initialize",
        action="store_true",
        help="Initialize repository index and exit (do not start server)"
    )
    mode_group.add_argument(
        "--transport",
        "-t",
        choices=["stdio", "http", "sse"],
        default=None,
        help="Transport type for server mode (default: stdio)"
    )
    
    # ... existing arguments ...
    
    args = parser.parse_args()
    
    # ... existing validation ...
    
    # Handle initialization mode
    if args.initialize:
        exit_code = asyncio.run(
            initialize_only(
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
                neo4j_database=neo4j_database,
                repository_path=str(repo_path.absolute()),
                project_name=project_name,
                file_patterns=file_patterns,
                verbose=args.verbose
            )
        )
        sys.exit(exit_code)
    
    # Otherwise run server (existing code)
    transport = args.transport or os.getenv("MCP_TRANSPORT", "stdio")
    # ... rest of existing server startup code ...
```

#### 1.3 Update Server to Use Shared Initializer
```python
# src/project_watch_mcp/server.py (modified sections only)

from .core.initializer import RepositoryInitializer, InitializationResult

def create_mcp_server(
    repository_monitor: RepositoryMonitor,
    neo4j_rag: Neo4jRAG,
    project_name: str,
) -> FastMCP:
    """Create an MCP server for repository monitoring and RAG."""
    
    mcp = FastMCP("project-watch-mcp", dependencies=["neo4j", "watchfiles", "pydantic"])
    
    # Create shared initializer
    initializer = RepositoryInitializer(
        repository_monitor=repository_monitor,
        neo4j_rag=neo4j_rag,
        project_name=project_name
    )
    
    @mcp.tool(
        annotations=ToolAnnotations(
            title="Initialize Repo",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        )
    )
    async def initialize_repository() -> ToolResult:
        """Initialize repository monitoring and perform initial scan."""
        try:
            # Use shared initializer
            result = await initializer.initialize(start_monitoring=True)
            
            return ToolResult(
                content=[TextContent(
                    type="text",
                    text=result.message
                )],
                structured_content={
                    "indexed": result.indexed,
                    "total": result.total,
                    "failed": len(result.failed),
                    "monitoring": result.monitoring_started
                }
            )
        except Exception as e:
            logger.error(f"Failed to initialize repository: {e}")
            raise ToolError(f"Failed to initialize repository: {e}")
    
    # ... rest of tools ...
```

#### 1.4 Simplify Hook Implementation
```python
#!/usr/bin/env python3
# .claude/hooks/session-start/session-start.py
"""
Session start hook for Project Watch MCP.
Delegates initialization to the CLI tool to avoid code duplication.
"""

import subprocess
import sys
import json
from pathlib import Path

def main():
    """Initialize repository using CLI."""
    project_root = Path(__file__).parent.parent.parent
    
    try:
        # Use CLI for initialization
        result = subprocess.run(
            ['uv', 'run', 'project-watch-mcp', '--initialize', '--verbose'],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        if result.returncode == 0:
            print("✅ Repository initialized successfully")
            print(result.stdout)
            return 0
        else:
            print("❌ Repository initialization failed", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return 1
            
    except subprocess.TimeoutExpired:
        print("❌ Initialization timed out after 60 seconds", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

### Phase 2: Future Enhancements (Optional)

#### 2.1 Implement Subcommands
```python
# Future: src/project_watch_mcp/cli.py with subcommands

def cli():
    parser = argparse.ArgumentParser(prog="project-watch-mcp")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize repository index')
    init_parser.add_argument('--verbose', '-v', action='store_true')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start MCP server')
    serve_parser.add_argument('--transport', choices=['stdio', 'http', 'sse'], default='stdio')
    serve_parser.add_argument('--port', type=int, default=8000)
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check index status')
    
    # Reindex command
    reindex_parser = subparsers.add_parser('reindex', help='Force full reindex')
    reindex_parser.add_argument('--clear', action='store_true', help='Clear existing index first')
    
    args = parser.parse_args()
    
    if args.command == 'init':
        return run_init(args)
    elif args.command == 'serve':
        return run_serve(args)
    elif args.command == 'status':
        return run_status(args)
    elif args.command == 'reindex':
        return run_reindex(args)
    else:
        parser.print_help()
        return 1
```

## Testing Strategy

### Unit Tests for Initializer
```python
# tests/test_initializer.py

import pytest
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from project_watch_mcp.core.initializer import RepositoryInitializer, InitializationResult


@pytest.mark.asyncio
async def test_initializer_success():
    """Test successful initialization."""
    # Mock dependencies
    mock_monitor = Mock()
    mock_monitor.scan_repository = AsyncMock(return_value=[
        Mock(path=Path("test.py"), language="python", size=100, last_modified="2024-01-01")
    ])
    mock_monitor.start = AsyncMock()
    
    mock_rag = Mock()
    mock_rag.index_file = AsyncMock()
    
    # Create initializer
    initializer = RepositoryInitializer(
        repository_monitor=mock_monitor,
        neo4j_rag=mock_rag,
        project_name="test-project"
    )
    
    # Run initialization
    result = await initializer.initialize(start_monitoring=True)
    
    # Assertions
    assert isinstance(result, InitializationResult)
    assert result.indexed == 1
    assert result.total == 1
    assert result.failed == []
    assert result.monitoring_started == True
    assert initializer.is_initialized == True
    
    # Verify calls
    mock_monitor.scan_repository.assert_called_once()
    mock_monitor.start.assert_called_once()
    mock_rag.index_file.assert_called_once()


@pytest.mark.asyncio
async def test_initializer_partial_failure():
    """Test initialization with some failed files."""
    # Setup mocks with mixed success/failure
    # ...
```

### Integration Tests
```python
# tests/integration/test_cli_init.py

import subprocess
import tempfile
from pathlib import Path


def test_cli_initialize_flag():
    """Test CLI --initialize flag."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test repository
        repo_path = Path(tmpdir) / "test-repo"
        repo_path.mkdir()
        (repo_path / "test.py").write_text("print('hello')")
        
        # Run CLI with initialize flag
        result = subprocess.run(
            ['python', '-m', 'project_watch_mcp', '--initialize', '--repository', str(repo_path)],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Initialized" in result.stdout
```

## Migration Checklist

- [ ] Create `src/project_watch_mcp/core/` directory
- [ ] Implement `RepositoryInitializer` class
- [ ] Add `--initialize` flag to CLI
- [ ] Update server.py to use shared initializer
- [ ] Simplify hook to use CLI
- [ ] Add unit tests for initializer
- [ ] Add integration tests for CLI
- [ ] Update documentation
- [ ] Test with real repository
- [ ] Verify hook works correctly

## Performance Considerations

1. **Memory Usage**: Large repositories may consume significant memory during indexing
   - Solution: Implement batch processing with configurable batch size
   
2. **Startup Time**: Initialization can be slow for large codebases
   - Solution: Add progress reporting and consider parallel processing
   
3. **Neo4j Connection Pool**: Ensure proper connection management
   - Solution: Use single driver instance with session management

## Security Considerations

1. **File Access**: Ensure repository boundaries are respected
2. **Neo4j Credentials**: Never log or expose credentials
3. **Error Messages**: Sanitize paths in error messages for production

## Rollback Plan

If issues arise:
1. Keep original server.py logic as fallback
2. Add feature flag to enable/disable new initializer
3. Maintain backward compatibility with existing MCP clients

## Success Metrics

- Zero code duplication between CLI, server, and hook
- Initialization time < 30 seconds for 1000 files
- Clean separation of concerns
- All tests passing
- No regression in existing functionality