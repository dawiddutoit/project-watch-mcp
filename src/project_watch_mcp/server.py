"""FastMCP server for repository monitoring with RAG capabilities."""

import logging
import os
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

try:
    from radon.complexity import cc_rank, cc_visit
    from radon.metrics import mi_visit

    RADON_AVAILABLE = True
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
    mcp = FastMCP("project-watch-mcp", dependencies=["neo4j", "watchfiles", "pydantic"])

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
            # Use the core initializer
            from .core.initializer import RepositoryInitializer

            # Get Neo4j configuration from environment (same as used in server setup)
            neo4j_uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
            neo4j_user = os.getenv("NEO4J_USER", "neo4j")
            neo4j_password = os.getenv("NEO4J_PASSWORD")
            neo4j_database = os.getenv("NEO4J_DB", os.getenv("NEO4J_DATABASE", "memory"))

            if not neo4j_password:
                raise ToolError("NEO4J_PASSWORD not set in environment")

            # Create initializer and run initialization
            # Use repository_monitor's path and project_name from closure
            initializer = RepositoryInitializer(
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
                neo4j_database=neo4j_database,
                repository_path=repository_monitor.repo_path,
                project_name=project_name,  # This comes from the create_mcp_server function parameter
            )

            async with initializer:
                result = await initializer.initialize(persistent_monitoring=True)

            # Format result for MCP tool
            return ToolResult(
                content=[TextContent(type="text", text=result.message)],
                structured_content=result.to_dict(),
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
            description="For semantic: Natural language description. For pattern: Exact text or regex",
        ),
        search_type: Literal["semantic", "pattern"] = Field(
            default="semantic",
            description="'semantic' for AI-powered conceptual search, 'pattern' for exact/regex matching",
        ),
        is_regex: bool = Field(
            default=False,
            description="Only for pattern search - treat query as regex (Python regex syntax)",
        ),
        limit: int = Field(
            default=10, description="Maximum results to return (default: 10, max: 100)"
        ),
        language: str | None = Field(
            default=None,
            description="Filter by programming language (e.g., 'python', 'javascript', 'typescript')",
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
            ..., description="Path to the file (relative from repo root or absolute within repo)"
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
            ..., description="Path to the file to refresh (relative from repo root or absolute)"
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
            title="Delete File",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=True,
        )
    )
    async def delete_file(
        file_path: str = Field(
            ...,
            description="Path to the file to delete from index (relative from repo root or absolute)",
        )
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

            # Check if file exists in index before deletion
            metadata = await neo4j_rag.get_file_metadata(path)

            if not metadata:
                result_text = f"File {file_path} not found in index"
                return ToolResult(
                    content=[TextContent(type="text", text=result_text)],
                    structured_content={
                        "status": "warning",
                        "message": "File not found in index",
                        "file": str(path),
                    },
                )

            chunks_count = metadata.get("chunk_count", 0)

            # Delete from Neo4j
            await neo4j_rag.delete_file(path)

            result_text = (
                f"Successfully deleted {file_path} from index ({chunks_count} chunks removed)"
            )

            return ToolResult(
                content=[TextContent(type="text", text=result_text)],
                structured_content={
                    "status": "success",
                    "message": "File removed from index",
                    "file": str(path),
                    "chunks_removed": chunks_count,
                },
            )

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
        file_path: str = Field(
            ...,
            description="Path to the Python file to analyze (relative from repo root or absolute)",
        ),
        include_metrics: bool = Field(
            default=True, description="Include additional metrics like maintainability index"
        ),
    ) -> ToolResult:
        """
        Calculate cyclomatic complexity for Python files.

        Analyzes Python code to determine its cyclomatic complexity, which measures
        the number of linearly independent paths through a program's source code.
        Higher complexity indicates more difficult code to understand and maintain.

        Examples:
            >>> # Analyze a single Python file
            >>> await analyze_complexity("src/main.py")
            {
                "file": "src/main.py",
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
                    },
                    {
                        "name": "simple_function",
                        "complexity": 2,
                        "rank": "A",
                        "line": 10,
                        "classification": "simple"
                    }
                ],
                "recommendations": [
                    "Consider refactoring 'complex_function' (complexity: 15)",
                    "2 functions have complexity > 10"
                ]
            }

            >>> # Analyze with basic metrics only
            >>> await analyze_complexity("tests/test_server.py", include_metrics=False)
            {
                "file": "tests/test_server.py",
                "summary": {
                    "total_complexity": 18,
                    "average_complexity": 2.0
                },
                "functions": [...]
            }

        Args:
            file_path: Path to the Python file to analyze
                - Relative paths: Interpreted from repository root
                - Absolute paths: Must be within repository
                - Must be a Python file (.py extension)
            include_metrics: Whether to include additional metrics like maintainability index

        Returns:
            ToolResult with complexity analysis:
            - file: Analyzed file path
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
            - recommendations: Suggestions for improvement

        Raises:
            ToolError: When analysis fails or file is not Python

        Notes:
            - Complexity ranks: A (1-5), B (6-10), C (11-20), D (21-30), E (31-40), F (41+)
            - Maintainability Index: >20 (maintainable), 10-20 (moderate), <10 (low)
            - Only analyzes Python files (.py extension)
            - Skips files that cannot be parsed as valid Python
            - Complex functions (>10) should be considered for refactoring
        """
        if not RADON_AVAILABLE:
            raise ToolError("Radon library not available. Please install it with: uv add radon")

        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = repository_monitor.repo_path / path

            # Check if file exists and is a Python file
            if not path.exists():
                raise ToolError(f"File {file_path} does not exist")

            if not str(path).endswith(".py"):
                raise ToolError(f"File {file_path} is not a Python file (.py extension required)")

            # Read file content
            content = path.read_text(encoding="utf-8")

            # Calculate cyclomatic complexity
            cc_results = cc_visit(content)

            # Prepare function complexity data
            functions = []
            total_complexity = 0

            for item in cc_results:
                complexity = item.complexity
                total_complexity += complexity

                # Determine classification
                if complexity <= 5:
                    classification = "simple"
                elif complexity <= 10:
                    classification = "moderate"
                elif complexity <= 20:
                    classification = "complex"
                else:
                    classification = "very-complex"

                functions.append(
                    {
                        "name": item.name,
                        "complexity": complexity,
                        "rank": cc_rank(complexity),
                        "line": item.lineno,
                        "classification": classification,
                        "type": item.__class__.__name__.lower(),  # 'function' or 'method'
                    }
                )

            # Sort functions by complexity (highest first)
            functions.sort(key=lambda x: x["complexity"], reverse=True)

            # Calculate average complexity
            avg_complexity = total_complexity / len(functions) if functions else 0

            # Prepare summary
            summary = {
                "total_complexity": total_complexity,
                "average_complexity": round(avg_complexity, 2),
                "function_count": len(functions),
            }

            # Add maintainability index if requested
            if include_metrics:
                mi_score = mi_visit(content, multi=False)
                summary["maintainability_index"] = round(mi_score, 2)

                # Determine grade based on MI score
                if mi_score >= 80:
                    grade = "A"
                elif mi_score >= 60:
                    grade = "B"
                elif mi_score >= 40:
                    grade = "C"
                elif mi_score >= 20:
                    grade = "D"
                else:
                    grade = "F"
                summary["complexity_grade"] = grade

            # Generate recommendations
            recommendations = []
            complex_functions = [f for f in functions if f["complexity"] > 10]
            very_complex_functions = [f for f in functions if f["complexity"] > 20]

            if very_complex_functions:
                for func in very_complex_functions[:3]:  # Top 3 most complex
                    recommendations.append(
                        f"Urgent: Refactor '{func['name']}' (complexity: {func['complexity']})"
                    )
            elif complex_functions:
                for func in complex_functions[:3]:  # Top 3 complex
                    recommendations.append(
                        f"Consider refactoring '{func['name']}' (complexity: {func['complexity']})"
                    )

            if complex_functions:
                recommendations.append(f"{len(complex_functions)} function(s) have complexity > 10")

            if avg_complexity > 10:
                recommendations.append(
                    f"High average complexity ({avg_complexity:.1f}). Consider breaking down functions"
                )

            if include_metrics and summary.get("maintainability_index", 100) < 20:
                recommendations.append(
                    "Low maintainability index. Code needs significant refactoring"
                )

            if not recommendations:
                recommendations.append("Code complexity is within acceptable limits")

            # Format result text
            result_text = f"Complexity Analysis for {file_path}:\n\n"
            result_text += "Summary:\n"
            result_text += f"  Total Complexity: {summary['total_complexity']}\n"
            result_text += f"  Average Complexity: {summary['average_complexity']}\n"
            result_text += f"  Functions Analyzed: {summary['function_count']}\n"

            if include_metrics:
                result_text += f"  Maintainability Index: {summary['maintainability_index']} (Grade: {summary['complexity_grade']})\n"

            result_text += "\nTop Complex Functions:\n"
            for func in functions[:5]:  # Show top 5
                result_text += f"  - {func['name']} (line {func['line']}): {func['complexity']} ({func['rank']}) - {func['classification']}\n"

            if recommendations:
                result_text += "\nRecommendations:\n"
                for rec in recommendations:
                    result_text += f"  • {rec}\n"

            return ToolResult(
                content=[TextContent(type="text", text=result_text)],
                structured_content={
                    "file": str(path.relative_to(repository_monitor.repo_path)),
                    "summary": summary,
                    "functions": functions[:20],  # Limit to top 20 for response size
                    "recommendations": recommendations,
                },
            )

        except SyntaxError as e:
            logger.error(f"Syntax error in file {file_path}: {e}")
            raise ToolError(f"Failed to parse Python file: {e}") from e
        except Exception as e:
            logger.error(f"Failed to analyze complexity: {e}")
            raise ToolError(f"Failed to analyze complexity: {e}") from e

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
