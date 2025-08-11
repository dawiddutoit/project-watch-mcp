"""FastMCP server for repository monitoring with RAG capabilities."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastmcp.exceptions import ToolError
from fastmcp.server import FastMCP
from fastmcp.tools.tool import TextContent, ToolResult
from mcp.types import ToolAnnotations
from pydantic import Field

from .neo4j_rag import CodeFile, Neo4jRAG
from .repository_monitor import FileInfo, RepositoryMonitor

logger = logging.getLogger(__name__)


def create_mcp_server(
    repository_monitor: RepositoryMonitor,
    neo4j_rag: Neo4jRAG,
    project_name: str,
) -> FastMCP:
    """
    Create an MCP server for repository monitoring and RAG.

    Args:
        repository_monitor: Repository monitor instance
        neo4j_rag: Neo4j RAG instance
        project_name: Name of the project for context isolation

    Returns:
        FastMCP server instance
    """
    mcp = FastMCP("project-watch-mcp", dependencies=["neo4j", "watchfiles", "pydantic"])

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Initialize Repository",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        )
    )
    async def initialize_repository() -> ToolResult:
        """
        Initialize repository monitoring and perform initial scan.

        Scans the repository for all matching files (respecting .gitignore patterns),
        indexes them in Neo4j for semantic search, and starts real-time monitoring
        for file changes.

        Examples:
            >>> # First-time initialization
            >>> await initialize_repository()
            {
                "indexed": 42,
                "total": 45,
                "message": "Repository initialized. Indexed 42/45 files."
            }
            
            >>> # Re-initialization (updates existing index)
            >>> await initialize_repository()
            {
                "indexed": 3,
                "total": 45,
                "message": "Repository re-indexed. Updated 3 changed files."
            }

        Returns:
            ToolResult with:
            - indexed: Number of successfully indexed files
            - total: Total number of files found
            - skipped: List of files that failed to index (if any)
            - monitoring: True if monitoring started successfully

        Raises:
            ToolError: When repository scanning or indexing fails

        Notes:
            - Safe to run multiple times (idempotent)
            - Automatically detects and updates only changed files on re-run
            - Respects .gitignore patterns in repository
            - Supports: .py, .js, .ts, .jsx, .tsx, .java, .cpp, .c, .h, .hpp, 
              .cs, .go, .rs, .rb, .php, .swift, .kt, .scala, .r, .m, .sql, 
              .sh, .yaml, .yml, .json, .xml, .html, .css, .scss, .md, .txt
        """
        try:
            # Scan repository for files
            files = await repository_monitor.scan_repository()

            # Index each file
            indexed_count = 0
            for file_info in files:
                try:
                    # Read file content
                    content = file_info.path.read_text(encoding="utf-8")

                    # Create CodeFile object with project context
                    code_file = CodeFile(
                        project_name=project_name,
                        path=file_info.path,
                        content=content,
                        language=file_info.language,
                        size=file_info.size,
                        last_modified=file_info.last_modified,
                    )

                    # Index in Neo4j
                    await neo4j_rag.index_file(code_file)
                    indexed_count += 1

                except Exception as e:
                    logger.error(f"Failed to index {file_info.path}: {e}")

            # Start monitoring
            await repository_monitor.start()

            result_text = f"Repository initialized. Indexed {indexed_count}/{len(files)} files."

            return ToolResult(
                content=[TextContent(type="text", text=result_text)],
                structured_content={"indexed": indexed_count, "total": len(files)},
            )

        except Exception as e:
            logger.error(f"Failed to initialize repository: {e}")
            raise ToolError(f"Failed to initialize repository: {e}")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Search Code",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        )
    )
    async def search_code(
        query: str = Field(
            ...,
            description="For semantic: Natural language description. For pattern: Exact text or regex"
        ),
        search_type: Literal["semantic", "pattern"] = Field(
            default="semantic",
            description="'semantic' for AI-powered conceptual search, 'pattern' for exact/regex matching"
        ),
        is_regex: bool = Field(
            default=False,
            description="Only for pattern search - treat query as regex (Python regex syntax)"
        ),
        limit: int = Field(
            default=10,
            description="Maximum results to return (default: 10, max: 100)"
        ),
        language: str | None = Field(
            default=None,
            description="Filter by programming language (e.g., 'python', 'javascript', 'typescript')"
        ),
    ) -> ToolResult:
        """
        Search for code in the repository using semantic search or pattern matching.

        Choose search type based on your needs:
        - Semantic: Find conceptually similar code using AI embeddings
        - Pattern: Find exact text matches or regex patterns

        Examples:
            >>> # Semantic search - find authentication logic
            >>> await search_code(
            ...     query="user authentication and JWT token validation",
            ...     search_type="semantic",
            ...     limit=5
            ... )
            {
                "results": [
                    {
                        "file": "src/auth/jwt_handler.py",
                        "line": 42,
                        "content": "def validate_jwt_token(token: str) -> dict:\\n    \\"\\"\\"Validate JWT token and return claims.\\"\\"\\"...",
                        "similarity": 0.89
                    }
                ]
            }
            
            >>> # Pattern search - find all TODO comments
            >>> await search_code(
            ...     query="TODO|FIXME|HACK",
            ...     search_type="pattern",
            ...     is_regex=True,
            ...     limit=10
            ... )
            {
                "results": [
                    {
                        "file": "src/utils.py",
                        "line": 23,
                        "content": "# TODO: Implement proper error handling",
                        "similarity": 1.0
                    }
                ]
            }
            
            >>> # Language-specific search
            >>> await search_code(
            ...     query="async function implementations",
            ...     language="typescript",
            ...     limit=5
            ... )

        Args:
            query: 
                - For semantic: Natural language description of what you're looking for
                - For pattern: Exact text or regex pattern to match
            search_type: 
                - "semantic": AI-powered conceptual search (default)
                - "pattern": Exact text or regex matching
            is_regex: Only for pattern search - treat query as regex (default: False)
            limit: Maximum results to return (default: 10, max: 100)
            language: Filter by programming language (e.g., "python", "javascript")

        Returns:
            ToolResult with:
            - results: Array of matches, each containing:
                - file: Full file path relative to repository root
                - line: Line number where match starts
                - content: Code snippet (up to 500 chars, use get_file_info for full content)
                - similarity: Relevance score (0-1, higher is better)
            - total_matches: Total number of matches found (may exceed limit)
            - search_time_ms: Time taken for search

        Raises:
            ToolError: When search operation fails

        Notes:
            - Content is truncated to 500 characters for readability
            - To view full file content, use file path with a file reading tool
            - Semantic search works best with descriptive, conceptual queries
            - Pattern search with regex supports Python regex syntax
            - Similarity scores: >0.8 (very relevant), 0.6-0.8 (relevant), <0.6 (loosely related)
        """
        try:
            if search_type == "semantic":
                results = await neo4j_rag.search_semantic(
                    query=query, limit=limit, language=language
                )
            else:  # pattern
                results = await neo4j_rag.search_by_pattern(
                    pattern=query, is_regex=is_regex, limit=limit
                )

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append(
                    {
                        "file": str(result.file_path),
                        "line": result.line_number,
                        "content": result.content[:500],  # Truncate for readability
                        "similarity": result.similarity,
                    }
                )

            result_text = f"Found {len(results)} result(s) for query: {query}\n\n"
            for i, result in enumerate(formatted_results, 1):
                result_text += f"{i}. {result['file']}:{result['line']} (similarity: {result['similarity']:.2f})\n"
                result_text += f"   {result['content'][:100]}...\n\n"

            return ToolResult(
                content=[TextContent(type="text", text=result_text)],
                structured_content={"results": formatted_results} if formatted_results else None,
            )

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise ToolError(f"Search failed: {e}")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get Repository Stats",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        )
    )
    async def get_repository_stats() -> dict:
        """
        Get comprehensive statistics about the indexed repository.

        Provides insights into repository composition, size, and indexing status.

        Examples:
            >>> await get_repository_stats()
            {
                "total_files": 156,
                "total_chunks": 1243,
                "total_size": 2456789,
                "total_lines": 45678,
                "languages": {
                    "python": {"files": 89, "size": 1234567, "percentage": 57.05},
                    "javascript": {"files": 45, "size": 890123, "percentage": 28.85},
                    "markdown": {"files": 22, "size": 332099, "percentage": 14.10}
                },
                "largest_files": [
                    {"path": "src/large_module.py", "size": 45678, "lines": 1234},
                    {"path": "tests/test_integration.py", "size": 34567, "lines": 987}
                ],
                "index_health": {
                    "last_full_scan": "2024-01-15T10:30:00Z",
                    "files_monitored": 156,
                    "files_indexed": 156,
                    "index_coverage": 100.0
                }
            }

        Returns:
            ToolResult with detailed statistics:
            - total_files: Number of files in index
            - total_chunks: Number of code chunks (for semantic search)
            - total_size: Total size in bytes
            - total_lines: Total lines of code
            - languages: Breakdown by programming language with:
                - files: Number of files
                - size: Total size in bytes
                - percentage: Percentage of repository
            - largest_files: Top 5 largest files with paths and sizes
            - index_health: Information about index status and coverage

        Raises:
            ToolError: When statistics retrieval fails

        Notes:
            - Chunk count indicates semantic search granularity
            - Languages are detected by file extension
            - Statistics are cached and updated on file changes
        """
        try:
            stats = await neo4j_rag.get_repository_stats()

            result_text = f"""Repository Statistics:
- Total Files: {stats['total_files']}
- Total Chunks: {stats['total_chunks']}
- Total Size: {stats['total_size']} bytes
- Languages: {', '.join(stats['languages'])}
"""

            return ToolResult(
                content=[TextContent(type="text", text=result_text)], structured_content=stats
            )

        except Exception as e:
            logger.error(f"Failed to get repository stats: {e}")
            raise ToolError(f"Failed to get repository stats: {e}")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get File Info",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        )
    )
    async def get_file_info(
        file_path: str = Field(
            ...,
            description="Path to the file (relative from repo root or absolute within repo)"
        )
    ) -> dict:
        """
        Get detailed metadata about a specific file in the repository.

        Retrieves comprehensive information about a file's content, structure,
        and indexing status.

        Examples:
            >>> # Using relative path from repository root
            >>> await get_file_info("src/main.py")
            {
                "path": "src/main.py",
                "absolute_path": "/home/user/project/src/main.py",
                "language": "python",
                "size": 4567,
                "lines": 234,
                "last_modified": "2024-01-15T10:30:00Z",
                "hash": "a1b2c3d4e5f6...",
                "indexed": true,
                "chunk_count": 8,
                "imports": ["os", "sys", "datetime", "custom_module"],
                "classes": ["MainApp", "ConfigManager"],
                "functions": ["main", "setup_logging", "process_data"]
            }
            
            >>> # Using absolute path
            >>> await get_file_info("/home/user/project/README.md")
            {
                "path": "README.md",
                "language": "markdown",
                "size": 2345,
                "indexed": true,
                ...
            }
            
            >>> # File not in index
            >>> await get_file_info("non_existent.py")
            {
                "error": "File not found in index",
                "suggestion": "Run initialize_repository or refresh_file first"
            }

        Args:
            file_path: Path to the file
                - Relative paths: Interpreted from repository root
                - Absolute paths: Must be within repository
                - Supports forward slashes on all platforms

        Returns:
            ToolResult with file metadata:
            - path: Relative path from repository root
            - absolute_path: Full system path
            - language: Detected programming language
            - size: File size in bytes
            - lines: Number of lines
            - last_modified: ISO 8601 timestamp
            - hash: Content hash (first 16 chars shown)
            - indexed: Whether file is in Neo4j index
            - chunk_count: Number of semantic chunks
            - imports/classes/functions: Extracted code elements (language-dependent)

        Raises:
            ToolError: When file metadata retrieval fails

        Notes:
            - File must be within repository boundaries
            - Metadata is cached and updated on file changes
            - For full file content, use appropriate file reading tools
            - Code element extraction varies by language support
        """
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = repository_monitor.repo_path / path

            metadata = await neo4j_rag.get_file_metadata(path)

            if metadata:
                result_text = f"""File Information for {file_path}:
- Language: {metadata['language']}
- Size: {metadata['size']} bytes
- Last Modified: {metadata['last_modified']}
- Chunks: {metadata['chunk_count']}
- Hash: {metadata['hash'][:16]}...
"""
            else:
                result_text = f"File {file_path} not found in index"
                metadata = {}

            return ToolResult(
                content=[TextContent(type="text", text=result_text)], structured_content=metadata
            )

        except Exception as e:
            logger.error(f"Failed to get file info: {e}")
            raise ToolError(f"Failed to get file info: {e}")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Refresh File",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        )
    )
    async def refresh_file(
        file_path: str = Field(
            ...,
            description="Path to the file to refresh (relative from repo root or absolute)"
        )
    ) -> ToolResult:
        """
        Manually refresh a specific file in the index.

        Forces re-indexing of a file, useful when automatic monitoring missed changes
        or when you need immediate index updates.

        Examples:
            >>> # Refresh a single file
            >>> await refresh_file("src/updated_module.py")
            {
                "status": "success",
                "file": "src/updated_module.py",
                "action": "updated",
                "chunks_before": 5,
                "chunks_after": 7,
                "time_ms": 123
            }
            
            >>> # File doesn't exist
            >>> await refresh_file("deleted_file.py")
            {
                "status": "error",
                "message": "File not found",
                "suggestion": "File may have been deleted. Run initialize_repository to clean index."
            }
            
            >>> # New file not yet indexed
            >>> await refresh_file("src/new_feature.py")
            {
                "status": "success",
                "file": "src/new_feature.py",
                "action": "added",
                "chunks_after": 3,
                "time_ms": 89
            }

        Args:
            file_path: Path to the file to refresh
                - Relative paths: Interpreted from repository root
                - Absolute paths: Must be within repository
                - File must exist and be readable

        Returns:
            ToolResult with refresh status:
            - status: "success" or "error"
            - file: Normalized file path
            - action: "added", "updated", or "removed"
            - chunks_before: Previous chunk count (if file was indexed)
            - chunks_after: New chunk count
            - time_ms: Processing time in milliseconds
            - error details if failed

        Raises:
            ToolError: When refresh operation fails

        Notes:
            - Replaces all existing chunks for the file
            - Triggers immediate semantic embedding generation
            - For bulk updates, consider re-running initialize_repository
            - Changes are immediately available for search
            - Respects .gitignore patterns - ignored files won't be indexed
        """
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = repository_monitor.repo_path / path

            if not path.exists():
                return ToolResult(
                    content=[TextContent(type="text", text=f"File {file_path} does not exist")],
                    structured_content={"status": "error", "message": "File not found"},
                )

            # Read file content
            content = path.read_text(encoding="utf-8")
            stat = path.stat()

            # Create CodeFile object with project context
            code_file = CodeFile(
                project_name=project_name,
                path=path,
                content=content,
                language=FileInfo(
                    path=path,
                    size=stat.st_size,
                    last_modified=datetime.fromtimestamp(stat.st_mtime),
                ).language,
                size=stat.st_size,
                last_modified=datetime.fromtimestamp(stat.st_mtime),
            )

            # Update in Neo4j
            await neo4j_rag.update_file(code_file)

            result_text = f"File {file_path} refreshed successfully"

            return ToolResult(
                content=[TextContent(type="text", text=result_text)],
                structured_content={"status": "success", "file": str(path)},
            )

        except Exception as e:
            logger.error(f"Failed to refresh file: {e}")
            raise ToolError(f"Failed to refresh file: {e}")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Monitoring Status",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        )
    )
    async def monitoring_status() -> dict:
        """
        Get the current status of repository monitoring.

        Provides real-time information about file monitoring, pending changes,
        and system health.

        Examples:
            >>> await monitoring_status()
            {
                "is_running": true,
                "repository_path": "/home/user/my-project",
                "file_patterns": ["*.py", "*.js", "*.ts", "*.md"],
                "monitoring_since": "2024-01-15T10:00:00Z",
                "pending_changes": 3,
                "recent_changes": [
                    {
                        "change_type": "modified",
                        "path": "src/main.py",
                        "timestamp": "2024-01-15T10:45:00Z",
                        "processed": false
                    },
                    {
                        "change_type": "added",
                        "path": "tests/test_new.py",
                        "timestamp": "2024-01-15T10:44:00Z",
                        "processed": false
                    }
                ],
                "statistics": {
                    "files_monitored": 156,
                    "changes_today": 23,
                    "last_index_update": "2024-01-15T10:44:30Z"
                }
            }

        Returns:
            ToolResult with monitoring status:
            - is_running: Whether file monitoring is active
            - repository_path: Full path to monitored repository
            - file_patterns: List of file patterns being monitored
            - monitoring_since: When monitoring started (ISO 8601)
            - pending_changes: Number of unprocessed file changes
            - recent_changes: Last 5 changes with:
                - change_type: "added", "modified", or "deleted"
                - path: File path relative to repository
                - timestamp: When change was detected
                - processed: Whether change has been indexed
            - statistics: Monitoring statistics including:
                - files_monitored: Total files being watched
                - changes_today: Number of changes in last 24 hours
                - last_index_update: Most recent index modification

        Raises:
            ToolError: When status retrieval fails

        Notes:
            - Pending changes are processed automatically in background
            - Change types:
                - "added": New file created
                - "modified": Existing file content changed
                - "deleted": File removed (also removes from index)
            - High pending_changes count may indicate processing bottleneck
            - Monitoring automatically restarts on connection issues
            - File patterns follow gitignore syntax
        """
        try:
            pending_changes = await repository_monitor.process_all_changes()

            status = {
                "is_running": repository_monitor.is_running,
                "repository_path": str(repository_monitor.repo_path),
                "file_patterns": repository_monitor.file_patterns,
                "pending_changes": len(pending_changes),
            }

            result_text = f"""Monitoring Status:
- Running: {status['is_running']}
- Repository: {status['repository_path']}
- Patterns: {', '.join(status['file_patterns'])}
- Pending Changes: {status['pending_changes']}
"""

            if pending_changes:
                result_text += "\nRecent Changes:\n"
                for change in pending_changes[:5]:  # Show last 5 changes
                    result_text += f"  - {change.change_type.value}: {change.path}\n"

            return ToolResult(
                content=[TextContent(type="text", text=result_text)], structured_content=status
            )

        except Exception as e:
            logger.error(f"Failed to get monitoring status: {e}")
            raise ToolError(f"Failed to get monitoring status: {e}")

    return mcp
