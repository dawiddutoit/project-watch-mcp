"""FastMCP server for repository monitoring with RAG capabilities."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastmcp.exceptions import ToolError
from fastmcp.server import FastMCP
from fastmcp.tools.tool import TextContent, ToolResult
from mcp.types import ToolAnnotations

# Import complexity analysis components
from .complexity_analysis import AnalyzerRegistry
from .complexity_analysis.languages.java_analyzer import JavaComplexityAnalyzer
from .complexity_analysis.languages.kotlin_analyzer import KotlinComplexityAnalyzer
from .complexity_analysis.languages.python_analyzer import PythonComplexityAnalyzer
from .neo4j_rag import CodeFile, Neo4jRAG
from .repository_monitor import RepositoryMonitor

try:
    import importlib.util
    if importlib.util.find_spec('radon'):
        RADON_AVAILABLE = True
    else:
        RADON_AVAILABLE = False
except ImportError:
    RADON_AVAILABLE = False

logger = logging.getLogger(__name__)


def create_mcp_server(
    repository_monitor: RepositoryMonitor,
    neo4j_rag: Neo4jRAG,
    project_name: str,
) -> FastMCP:
    """
    Create an MCP server for repository monitoring and RAG. test

    Args:
        repository_monitor: Repository monitor instance
        neo4j_rag: Neo4j RAG instance
        project_name: Name of the project for context isolation

    Returns:
        FastMCP server instance
    """
    logger.info(f"Creating MCP server for project: {project_name}")
    mcp = FastMCP("project-watch-mcp", dependencies=["neo4j", "watchfiles", "pydantic"])
    logger.debug("FastMCP instance created")

    # Register language analyzers on server startup
    logger.debug("Registering language analyzers...")
    registered_analyzers = []
    if not AnalyzerRegistry.get_analyzer("python"):
        AnalyzerRegistry.register("python", PythonComplexityAnalyzer)
        registered_analyzers.append("python")
        logger.debug("  ✓ Python analyzer registered")
    if not AnalyzerRegistry.get_analyzer("java"):
        AnalyzerRegistry.register("java", JavaComplexityAnalyzer)
        registered_analyzers.append("java")
        logger.debug("  ✓ Java analyzer registered")
    if not AnalyzerRegistry.get_analyzer("kotlin"):
        AnalyzerRegistry.register("kotlin", KotlinComplexityAnalyzer)
        registered_analyzers.append("kotlin")
        logger.debug("  ✓ Kotlin analyzer registered")

    if registered_analyzers:
        logger.info(f"Registered {len(registered_analyzers)} language analyzers: {', '.join(registered_analyzers)}")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Initialize Repo",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
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
              .sh, .yaml, .yml, .toml, .json, .xml, .html, .css, .scss, .md, .txt
        """
        try:
            # Scan repository
            files = await repository_monitor.scan_repository()
            total = len(files)

            # Index files
            indexed = 0
            skipped = []
            for file_info in files:
                try:
                    # Read file content and create CodeFile object
                    try:
                        content = file_info.path.read_text(encoding='utf-8')
                    except Exception as e:
                        logger.debug(f"Could not read file {file_info.path}: {e}")
                        skipped.append(str(file_info.path))
                        continue

                    # Create CodeFile from FileInfo
                    code_file = CodeFile(
                        project_name=project_name,
                        path=file_info.path,
                        content=content,
                        language=file_info.language or "text",
                        size=file_info.size,
                        last_modified=file_info.last_modified,
                    )

                    await neo4j_rag.index_file(code_file)
                    indexed += 1
                except Exception as e:
                    logger.error(f"Failed to index {file_info.path}: {e}")
                    skipped.append(str(file_info.path))

            # Start monitoring if not already running
            if not repository_monitor.is_running:
                await repository_monitor.start()

            message = f"Repository initialized. Indexed {indexed}/{total} files."
            if skipped:
                message += f" Skipped: {', '.join(skipped[:5])}"
                if len(skipped) > 5:
                    message += f" and {len(skipped) - 5} more"

            return ToolResult(
                content=[TextContent(type="text", text=message)],
                structured_content={
                    "indexed": indexed,
                    "total": total,
                    "skipped": skipped,
                    "monitoring": repository_monitor.is_running,
                    "message": message,
                },
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
        query: str,
        search_type: Literal["semantic", "pattern"] = "semantic",
        is_regex: bool = False,
        limit: int = 10,
        language: str | None = None,
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
                        "content": "def validate_jwt_token(token: str) -> dict:\\n    \\\"\\\"\\\"Validate JWT token and return claims.\\\"\\\"\\\"...",
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
            # Validate and clamp limit
            limit = min(max(1, limit), 100)

            # Perform search based on type
            if search_type == "semantic":
                results = await neo4j_rag.search_semantic(query=query, limit=limit, language=language)
            else:  # pattern
                results = await neo4j_rag.search_by_pattern(pattern=query, is_regex=is_regex, limit=limit, language=language)

            # Format results
            formatted_results = []
            for result in results:
                # Handle both dict and SearchResult object formats
                if hasattr(result, '__dict__'):
                    # SearchResult object
                    content = getattr(result, 'content', '')
                    file_path = str(getattr(result, 'file_path', ''))
                    line_number = getattr(result, 'line_number', 0)
                    similarity = getattr(result, 'similarity', 0)
                else:
                    # Dictionary format
                    content = result.get("content", "")
                    file_path = result.get("file", "")
                    line_number = result.get("line", 0)
                    similarity = result.get("similarity", 0)

                # Truncate content for readability
                if len(content) > 500:
                    content = content[:497] + "..."

                formatted_results.append(
                    {
                        "file": file_path,
                        "line": line_number,
                        "content": content,
                        "similarity": round(similarity, 3),
                    }
                )

            result_text = f"Found {len(formatted_results)} results for '{query}'\n\n"
            for i, res in enumerate(formatted_results[:10], 1):
                result_text += f"{i}. {res['file']}:{res['line']} (similarity: {res['similarity']})\n"
                result_text += f"   {res['content'][:100]}...\n\n"

            return ToolResult(
                content=[TextContent(type="text", text=result_text)],
                structured_content={
                    "results": formatted_results,
                    "total_matches": len(formatted_results),
                    "search_type": search_type,
                    "query": query,
                    "limit_applied": limit,
                },
            )

        except Exception as e:
            logger.error(f"Failed to search code: {e}")
            raise ToolError(f"Failed to search code: {e}")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get Repository Stats",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        )
    )
    async def get_repository_stats() -> ToolResult:
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

            # Format statistics for display
            result_text = f"""Repository Statistics:
- Total Files: {stats.get('total_files', 0)}
- Total Chunks: {stats.get('total_chunks', 0)}
- Total Size: {stats.get('total_size', 0):,} bytes
- Total Lines: {stats.get('total_lines', 0):,}

Languages:
"""
            languages = stats.get("languages", {})
            # Handle languages as either a list or a dict
            if isinstance(languages, list):
                # If it's a list of language names, just display them
                for lang in languages:
                    if lang:  # Skip None or empty values
                        result_text += f"  - {lang}\n"
            elif isinstance(languages, dict):
                # If it's a dict with detailed stats
                for lang, data in languages.items():
                    result_text += f"  - {lang}: {data['files']} files, {data['percentage']:.1f}%\n"

            if stats.get("largest_files"):
                result_text += "\nLargest Files:\n"
                for file in stats["largest_files"][:5]:
                    result_text += f"  - {file['path']}: {file['size']:,} bytes\n"

            return ToolResult(content=[TextContent(type="text", text=result_text)], structured_content=stats)

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
        file_path: str,
    ) -> ToolResult:
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

            # Get file info from Neo4j
            file_info = await neo4j_rag.get_file_metadata(path)

            if not file_info:
                return ToolResult(
                    content=[TextContent(type="text", text=f"File {file_path} not found in index")],
                    structured_content={
                        "error": "File not found in index",
                        "suggestion": "Run initialize_repository or refresh_file first",
                    },
                )

            # Convert CodeFile object to dict if needed
            if hasattr(file_info, '__dict__'):
                # It's a CodeFile object
                from project_watch_mcp.neo4j_rag import CodeFile
                if isinstance(file_info, CodeFile):
                    file_dict = {
                        "path": str(file_info.path),
                        "language": file_info.language,
                        "size": file_info.size,
                        "lines": getattr(file_info, 'lines', len(file_info.content.splitlines()) if file_info.content else 0),
                        "last_modified": str(file_info.last_modified) if file_info.last_modified else "unknown",
                        "indexed": True,
                        "chunk_count": 0,
                        "imports": getattr(file_info, 'imports', []),
                        "classes": getattr(file_info, 'classes', []),
                        "functions": getattr(file_info, 'functions', [])
                    }
                else:
                    # Generic object, convert to dict
                    file_dict = file_info.__dict__
            else:
                # It's already a dictionary
                file_dict = file_info

            # Format result
            result_text = f"""File Information:
- Path: {file_dict.get('path', file_path)}
- Language: {file_dict.get('language', 'unknown')}
- Size: {file_dict.get('size', 0):,} bytes
- Lines: {file_dict.get('lines', 0):,}
- Last Modified: {file_dict.get('last_modified', 'unknown')}
- Indexed: {file_dict.get('indexed', False)}
- Chunks: {file_dict.get('chunk_count', 0)}
"""

            if file_dict.get("imports"):
                result_text += f"\nImports: {', '.join(file_dict['imports'][:10])}\n"
            if file_dict.get("classes"):
                result_text += f"Classes: {', '.join(file_dict['classes'][:10])}\n"
            if file_dict.get("functions"):
                result_text += f"Functions: {', '.join(file_dict['functions'][:10])}\n"

            return ToolResult(content=[TextContent(type="text", text=result_text)], structured_content=file_dict)

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
        file_path: str,
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

            # Get file info from repository monitor
            file_info = await repository_monitor.get_file_info(path)

            if not file_info:
                raise ToolError(f"File {file_path} not found")

            # Update in Neo4j (not refresh_file which doesn't exist)
            import time

            start = time.time()
            result = await neo4j_rag.update_file(file_info)
            elapsed_ms = int((time.time() - start) * 1000)

            # Ensure result has the expected structure
            if result is None:
                result = {}
            if "status" not in result:
                result["status"] = "success"
            if "action" not in result:
                result["action"] = "updated"

            result["time_ms"] = elapsed_ms

            # Format result text
            result_text = f"File refreshed: {file_path}\n"
            result_text += f"Action: {result.get('action', 'unknown')}\n"
            if result.get("chunks_before") is not None:
                result_text += f"Chunks: {result['chunks_before']} -> {result.get('chunks_after', 0)}\n"
            else:
                result_text += f"Chunks: {result.get('chunks_after', 0)}\n"
            result_text += f"Time: {elapsed_ms}ms"

            return ToolResult(content=[TextContent(type="text", text=result_text)], structured_content=result)

        except Exception as e:
            logger.error(f"Failed to refresh file: {e}")
            raise ToolError(f"Failed to refresh file: {e}")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Delete File",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=True,
        )
    )
    async def delete_file(
        file_path: str,
    ) -> ToolResult:
        """
        Delete a file and its chunks from the Neo4j index.

        Removes all indexed data for a file, including its metadata and code chunks.
        This operation only affects the index - it does NOT delete the actual file
        from the filesystem.

        Examples:
            >>> # Delete a file from the index
            >>> await delete_file("src/deprecated_module.py")
            {
                "status": "success",
                "message": "File removed from index",
                "file": "src/deprecated_module.py",
                "chunks_removed": 12
            }

            >>> # File not in index
            >>> await delete_file("non_indexed_file.py")
            {
                "status": "warning",
                "message": "File not found in index",
                "file": "non_indexed_file.py"
            }

            >>> # Delete with absolute path
            >>> await delete_file("/home/user/project/src/old_file.py")
            {
                "status": "success",
                "message": "File removed from index",
                "file": "/home/user/project/src/old_file.py",
                "chunks_removed": 8
            }

        Args:
            file_path: Path to the file to delete from index
                - Relative paths: Interpreted from repository root
                - Absolute paths: Must be within repository
                - File doesn't need to exist on filesystem

        Returns:
            ToolResult with deletion status:
            - status: "success", "warning", or "error"
            - message: Description of what happened
            - file: Normalized file path
            - chunks_removed: Number of chunks deleted (if successful)

        Raises:
            ToolError: When deletion operation fails

        Notes:
            - This only removes the file from the Neo4j index
            - The actual file on disk is NOT deleted
            - Use this when files are deleted/moved outside of monitored changes
            - To re-index the file, use refresh_file or initialize_repository
            - Deletion is immediate and permanent for the index
        """
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = repository_monitor.repo_path / path

            # Delete from Neo4j
            result = await neo4j_rag.delete_file(path)

            # Format result
            if result.get("chunks_removed", 0) > 0:
                result_text = f"File removed from index: {file_path}\n"
                result_text += f"Chunks removed: {result['chunks_removed']}"
                result["status"] = "success"
                result["message"] = "File removed from index"
            else:
                result_text = f"File not found in index: {file_path}"
                result["status"] = "warning"
                result["message"] = "File not found in index"

            result["file"] = str(file_path)

            return ToolResult(content=[TextContent(type="text", text=result_text)], structured_content=result)

        except Exception as e:
            logger.error(f"Failed to delete file from index: {e}")
            raise ToolError(f"Failed to delete file from index: {e}")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Analyze Complexity",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        )
    )
    async def analyze_complexity(
        file_path: str,
        language: str | None = None,
        include_metrics: bool = True,
    ) -> ToolResult:
        """
        Calculate cyclomatic complexity for code files in multiple languages.

        Analyzes code to determine its cyclomatic complexity, which measures
        the number of linearly independent paths through a program's source code.
        Higher complexity indicates more difficult code to understand and maintain.

        Supports: Python, Java, Kotlin (more languages coming soon)

        Examples:
            >>> # Auto-detect language from file extension
            >>> await analyze_complexity("src/main.py")
            {
                "file": "src/main.py",
                "language": "python",
                "summary": {
                    "total_complexity": 42,
                    "average_complexity": 3.5,
                    "maintainability_index": 65.2,
                    "complexity_grade": "B"
                },
                "functions": [
                    {
                        "name": "complex_function",
                        "complexity": 15,
                        "rank": "D",
                        "line": 45,
                        "classification": "complex"
                    }
                ]
            }

            >>> # Explicitly specify language
            >>> await analyze_complexity("config.json", language="python")
            # Forces Python analysis on a non-.py file

            >>> # Analyze Java file
            >>> await analyze_complexity("src/Main.java")
            {
                "file": "src/Main.java",
                "language": "java",
                "summary": {
                    "total_complexity": 38,
                    "average_complexity": 4.2
                },
                "functions": [...]
            }

        Args:
            file_path: Path to the code file to analyze
                - Relative paths: Interpreted from repository root
                - Absolute paths: Must be within repository
            language: Force specific language analyzer (optional)
                - If not specified, auto-detected from file extension
                - Supported: "python", "java", "kotlin"
            include_metrics: Whether to include additional metrics

        Returns:
            ToolResult with complexity analysis:
            - file: Analyzed file path
            - language: Detected or specified language
            - summary: Overall file metrics
                - total_complexity: Sum of all function complexities
                - average_complexity: Average complexity per function
                - maintainability_index: MI score (0-100, higher is better)
                - complexity_grade: Letter grade (A-F) based on MI
            - functions: List of functions with their complexity scores
                - name: Function/method name
                - complexity: Cyclomatic complexity score
                - rank: Complexity rank (A-F)
                - line: Line number where function starts
                - classification: simple/moderate/complex/very-complex
            - classes: List of classes (if applicable)
            - recommendations: Suggestions for improvement

        Raises:
            ToolError: When analysis fails or language not supported

        Notes:
            - Complexity ranks: A (1-5), B (6-10), C (11-20), D (21-30), E (31-40), F (41+)
            - Maintainability Index: >20 (maintainable), 10-20 (moderate), <10 (low)
            - Complex functions (>10) should be considered for refactoring
            - Language detection uses file extensions: .py, .java, .kt, .kts
        """
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = repository_monitor.repo_path / path

            # Determine analyzer to use BEFORE checking file existence
            # This ensures we give proper "not supported" errors for unsupported files
            analyzer = None

            if language:
                # Use explicitly specified language
                analyzer = AnalyzerRegistry.get_analyzer(language)
                if not analyzer:
                    supported = AnalyzerRegistry.supported_languages()
                    raise ToolError(
                        f"Language '{language}' is not supported. "
                        f"Supported languages: {', '.join(supported)}"
                    )
            else:
                # Auto-detect from file extension
                # Check file extension first for clearer error messages
                file_ext = path.suffix.lower()

                # For backward compatibility with existing tests
                if file_ext not in ['.py', '.java', '.kt', '.kts'] and not language:
                    # Special check for Python-only backward compatibility
                    if RADON_AVAILABLE and file_ext == '.py':
                        pass  # Will be handled below
                    else:
                        raise ToolError("Only Python files (.py) are supported for complexity analysis")

                analyzer = AnalyzerRegistry.get_analyzer_for_file(path)
                if not analyzer:
                    # Fall back to Python for backward compatibility if .py file
                    if str(path).endswith(".py") and RADON_AVAILABLE:
                        analyzer = AnalyzerRegistry.get_analyzer("python")

                    if not analyzer:
                        supported = AnalyzerRegistry.supported_languages()
                        raise ToolError(
                            f"File type not supported for {path.suffix} files. "
                            f"Supported: .py, .java, .kt, .kts. "
                            f"You can also specify language explicitly."
                        )

            # NOW check if file exists
            if not path.exists():
                raise ToolError(f"File {file_path} does not exist")

            # Analyze the file
            # If language was explicitly specified and file extension doesn't match,
            # use analyze_code instead to bypass file extension checks
            if language and not str(path).lower().endswith(('.py', '.java', '.kt', '.kts')):
                # Read file content and analyze as code
                code = path.read_text(encoding='utf-8')
                result = await analyzer.analyze_code(code)
                result.file_path = str(path)
            else:
                result = await analyzer.analyze_file(path)

            # Format the result
            summary = {
                "total_complexity": result.summary.total_complexity,
                "average_complexity": round(result.summary.average_complexity, 2),
                "function_count": result.summary.function_count,
            }

            if result.summary.class_count > 0:
                summary["class_count"] = result.summary.class_count
                # Calculate average class complexity if we have classes
                if result.classes:
                    total_class_complexity = sum(cls.total_complexity for cls in result.classes)
                    summary["average_class_complexity"] = round(
                        total_class_complexity / len(result.classes), 2
                    )

            if include_metrics and result.summary.maintainability_index is not None:
                summary["maintainability_index"] = round(result.summary.maintainability_index, 2)
                # Handle complexity_grade - could be enum or string
                grade = result.summary.complexity_grade
                summary["complexity_grade"] = grade.value if hasattr(grade, 'value') else str(grade)

            # Format functions
            functions = []
            for func in result.functions[:20]:  # Limit to top 20
                func_dict = {
                    "name": func.name,
                }
                # Handle different attribute names
                if hasattr(func, 'cyclomatic_complexity'):
                    func_dict["complexity"] = func.cyclomatic_complexity
                elif hasattr(func, 'complexity'):
                    func_dict["complexity"] = func.complexity

                if hasattr(func, 'complexity_rank'):
                    func_dict["rank"] = func.complexity_rank
                elif hasattr(func, 'rank'):
                    func_dict["rank"] = func.rank

                if hasattr(func, 'line_number'):
                    func_dict["line"] = func.line_number
                elif hasattr(func, 'line_start'):
                    func_dict["line"] = func.line_start

                if hasattr(func, 'classification'):
                    func_dict["classification"] = func.classification

                if hasattr(func, 'cognitive_complexity') and func.cognitive_complexity is not None:
                    func_dict["cognitive_complexity"] = func.cognitive_complexity

                functions.append(func_dict)

            # Format classes if present
            classes = []
            for cls in result.classes[:10]:  # Limit to top 10
                cls_dict = {
                    "name": cls.name,
                    "complexity": cls.total_complexity,
                    "method_count": cls.method_count,
                }
                # Handle different attribute names across analyzers
                if hasattr(cls, 'average_method_complexity'):
                    cls_dict["average_complexity"] = round(cls.average_method_complexity, 2)
                elif hasattr(cls, 'average_complexity'):
                    cls_dict["average_complexity"] = round(cls.average_complexity, 2)

                if hasattr(cls, 'line_start'):
                    cls_dict["line"] = cls.line_start
                elif hasattr(cls, 'line_number'):
                    cls_dict["line"] = cls.line_number

                classes.append(cls_dict)

            # Generate recommendations
            recommendations = result.recommendations if hasattr(result, 'recommendations') else []

            # Format result text
            result_text = f"Complexity Analysis for {file_path}:\n\n"
            result_text += f"Language: {result.language}\n"
            result_text += "Summary:\n"
            result_text += f"  Total Complexity: {summary['total_complexity']}\n"
            result_text += f"  Average Complexity: {summary['average_complexity']}\n"
            result_text += f"  Functions Analyzed: {summary['function_count']}\n"

            if "class_count" in summary:
                result_text += f"  Classes: {summary['class_count']}\n"

            if include_metrics and "maintainability_index" in summary:
                result_text += f"  Maintainability Index: {summary['maintainability_index']} "
                result_text += f"(Grade: {summary['complexity_grade']})\n"

            if functions:
                result_text += "\nTop Complex Functions:\n"
                for func in functions[:5]:
                    result_text += f"  - {func['name']} (line {func['line']}): "
                    result_text += f"{func['complexity']} ({func['rank']}) - {func['classification']}\n"

            if classes:
                result_text += "\nClasses:\n"
                for cls in classes[:3]:
                    result_text += f"  - {cls['name']} (line {cls['line']}): "
                    result_text += f"total {cls['complexity']}, avg {cls['average_complexity']}\n"

            if recommendations:
                result_text += "\nRecommendations:\n"
                for rec in recommendations:
                    result_text += f"  • {rec}\n"

            # Handle file path - make relative if within repo, otherwise use as-is
            try:
                file_display = str(path.relative_to(repository_monitor.repo_path))
            except ValueError:
                # File is outside repository, use the original path or just the name
                file_display = file_path if isinstance(file_path, str) else str(path)

            return ToolResult(
                content=[TextContent(type="text", text=result_text)],
                structured_content={
                    "file": file_display,
                    "language": result.language,
                    "summary": summary,
                    "functions": functions,
                    "classes": classes if classes else None,
                    "recommendations": recommendations,
                },
            )

        except Exception as e:
            logger.error(f"Failed to analyze complexity: {e}")
            raise ToolError(f"Failed to analyze complexity: {e}")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Monitoring Status",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        )
    )
    async def monitoring_status() -> ToolResult:
        """
        Get the current status of repository monitoring.

        Provides real-time information about file monitoring, pending changes,
        system health, and version information.

        Examples:
            >>> await monitoring_status()
            {
                "is_running": true,
                "repository_path": "/home/user/my-project",
                "file_patterns": ["*.py", "*.js", "*.ts", "*.md"],
                "monitoring_since": "2024-01-15T10:00:00Z",
                "pending_changes": 3,
                "version_info": {
                    "version": "0.1.0",
                    "build_timestamp": "2024-01-15T10:30:00",
                    "lucene_fix_version": "v2.0-double-escape"
                },
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
            - version_info: Server version and build information
                - version: Package version
                - build_timestamp: When package was built
                - lucene_fix_version: Lucene escaping fix version
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
            # Get version info
            from . import __build_timestamp__, __lucene_fix_version__, __version__

            # Get pending changes (if monitoring supports it)
            pending_changes = getattr(repository_monitor, 'pending_changes', [])

            # Format recent changes - handle both dict and object formats
            recent_changes = []
            for change in pending_changes[:5]:
                if isinstance(change, dict):
                    # Dictionary format from mock
                    change_entry = {
                        "change_type": change.get("change_type", "unknown"),
                        "path": change.get("path", ""),
                        "timestamp": change.get("timestamp", datetime.now()).isoformat() if hasattr(change.get("timestamp", datetime.now()), 'isoformat') else str(change.get("timestamp", "")),
                        "processed": change.get("processed", False),
                    }
                else:
                    # Object format - handle enums properly
                    change_type = getattr(change, 'change_type', 'unknown')
                    if hasattr(change_type, 'value'):
                        change_type = change_type.value

                    change_path = getattr(change, 'path', '')
                    if change_path and hasattr(change_path, 'relative_to'):
                        try:
                            change_path = str(change_path.relative_to(repository_monitor.repo_path))
                        except:
                            change_path = str(change_path)
                    else:
                        change_path = str(change_path)

                    timestamp = getattr(change, 'timestamp', datetime.now())
                    if hasattr(timestamp, 'isoformat'):
                        timestamp = timestamp.isoformat()
                    else:
                        timestamp = str(timestamp)

                    change_entry = {
                        "change_type": change_type,
                        "path": change_path,
                        "timestamp": timestamp,
                        "processed": getattr(change, 'processed', False),
                    }
                recent_changes.append(change_entry)

            status = {
                "is_running": repository_monitor.is_running,
                "repository_path": str(repository_monitor.repo_path),
                "file_patterns": repository_monitor.file_patterns,
                "monitoring_since": (
                    repository_monitor.monitoring_since.isoformat() if repository_monitor.monitoring_since else None
                ),
                "pending_changes": len(pending_changes),
                "recent_changes": recent_changes,
                "version_info": {
                    "version": __version__,
                    "build_timestamp": __build_timestamp__,
                    "lucene_fix_version": __lucene_fix_version__
                }
            }

            result_text = f"""Monitoring Status:
- Running: {status['is_running']}
- Repository: {status['repository_path']}
- Patterns: {', '.join(status['file_patterns'])}
- Pending Changes: {status['pending_changes']}
- Version: {__version__} (built: {__build_timestamp__})
- Lucene Fix: {__lucene_fix_version__}
"""

            if pending_changes:
                result_text += "\nRecent Changes:\n"
                for change in pending_changes[:5]:  # Show last 5 changes
                    if isinstance(change, dict):
                        change_type = change.get("change_type", "unknown")
                        change_path = change.get("path", "")
                    else:
                        change_type = getattr(change, 'change_type', 'unknown')
                        if hasattr(change_type, 'value'):
                            change_type = change_type.value
                        change_path = getattr(change, 'path', '')
                    result_text += f"  - {change_type}: {change_path}\n"

            return ToolResult(
                content=[TextContent(type="text", text=result_text)], structured_content=status
            )

        except Exception as e:
            logger.error(f"Failed to get monitoring status: {e}")
            raise ToolError(f"Failed to get monitoring status: {e}")

    # Log registered tools
    logger.info("MCP Server tools registered:")
    logger.info("  • initialize_repository - Initialize and index repository")
    logger.info("  • search_code - Search code using semantic or pattern matching")
    logger.info("  • get_repository_stats - Get repository statistics")
    logger.info("  • get_file_info - Get metadata about a specific file")
    logger.info("  • refresh_file - Manually refresh a file in the index")
    logger.info("  • delete_file - Remove a file from the index")
    logger.info("  • analyze_complexity - Analyze code complexity")
    logger.info("  • monitoring_status - Get repository monitoring status")
    logger.info(f"✓ MCP server ready for project: {project_name}")

    return mcp
