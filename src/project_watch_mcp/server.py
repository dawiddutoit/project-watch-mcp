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
) -> FastMCP:
    """
    Create an MCP server for repository monitoring and RAG.

    Args:
        repository_monitor: Repository monitor instance
        neo4j_rag: Neo4j RAG instance

    Returns:
        FastMCP server instance
    """
    mcp = FastMCP(
        "project-watch-mcp", dependencies=["neo4j", "watchfiles", "pydantic"]
    )

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Initialize Repository",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        )
    )
    async def initialize_repository() -> str:
        """
        Initialize repository monitoring and perform initial scan.
        Indexes all matching files in the repository.
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

                    # Create CodeFile object
                    code_file = CodeFile(
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
        query: str = Field(..., description="Search query"),
        search_type: Literal["semantic", "pattern"] = Field(
            default="semantic", description="Type of search to perform"
        ),
        is_regex: bool = Field(
            default=False, description="Whether pattern is a regex (for pattern search)"
        ),
        limit: int = Field(default=10, description="Maximum number of results"),
        language: str | None = Field(default=None, description="Filter by programming language"),
    ) -> list[dict]:
        """
        Search for code in the repository using semantic search or pattern matching.

        Args:
            query: Search query or pattern
            search_type: "semantic" for embedding-based search, "pattern" for text/regex
            is_regex: Whether the pattern is a regular expression
            limit: Maximum number of results to return
            language: Optional language filter

        Returns:
            List of search results with file paths, content, and relevance scores
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
                structured_content=formatted_results,
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
        Get statistics about the indexed repository.

        Returns:
            Repository statistics including file count, languages, and size
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
    async def get_file_info(file_path: str = Field(..., description="Path to the file")) -> dict:
        """
        Get metadata about a specific file in the repository.

        Args:
            file_path: Path to the file (relative or absolute)

        Returns:
            File metadata including language, size, and indexing info
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
        file_path: str = Field(..., description="Path to the file to refresh")
    ) -> str:
        """
        Manually refresh a specific file in the index.

        Args:
            file_path: Path to the file to refresh

        Returns:
            Status message
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

            # Create CodeFile object
            code_file = CodeFile(
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

        Returns:
            Monitoring status including running state and pending changes
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
