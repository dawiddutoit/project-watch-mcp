"""Test suite for project context persistence and requirements.

This module tests that project_name is properly required, persisted,
and included in all operations and results.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.project_watch_mcp.neo4j_rag import CodeChunk, CodeFile, Neo4jRAG, SearchResult
from src.project_watch_mcp.repository_monitor import RepositoryMonitor
from src.project_watch_mcp.server import create_mcp_server
from src.project_watch_mcp.utils.embedding import MockEmbeddingsProvider


class TestProjectContextPersistence:
    """Test that project context is properly persisted throughout the system."""

    @pytest.fixture
    def mock_driver_with_validation(self):
        """Create a mock Neo4j driver that validates project_name presence."""
        driver = AsyncMock()
        driver.verify_connectivity = AsyncMock()
        driver.close = AsyncMock()

        # Storage for indexed data
        driver._storage = {}

        async def execute_query_with_validation(query, params=None, *args, **kwargs):
            """Execute query with strict project_name validation."""
            # Check that project_name is always provided for data operations
            if any(keyword in query for keyword in ["MERGE", "CREATE", "MATCH", "DELETE"]):
                if not params or "project_name" not in params:
                    raise ValueError(f"project_name is required for query: {query[:50]}...")

            project_name = params.get("project_name") if params else None

            # Initialize project storage
            if project_name and project_name not in driver._storage:
                driver._storage[project_name] = {"files": {}, "chunks": [], "metadata": {}}

            # Handle different operations
            if "MERGE (f:CodeFile" in query:
                # Store file with project context
                if project_name:
                    driver._storage[project_name]["files"][params["path"]] = {
                        "project_name": project_name,
                        "language": params.get("language"),
                        "size": params.get("size"),
                        "hash": params.get("hash"),
                        "last_modified": params.get("last_modified"),
                    }
                return MagicMock(records=[])

            elif "CREATE (c:CodeChunk" in query:
                # Store chunk with project context
                if project_name:
                    driver._storage[project_name]["chunks"].append(
                        {
                            "project_name": project_name,
                            "file_path": params.get("file_path"),
                            "content": params.get("content"),
                            "start_line": params.get("start_line"),
                            "end_line": params.get("end_line"),
                            "embedding": params.get("embedding"),
                        }
                    )
                return MagicMock(records=[])

            elif "MATCH (f:CodeFile" in query and params and "path" in params:
                # Return file metadata
                if project_name and params["path"] in driver._storage[project_name]["files"]:
                    file_data = driver._storage[project_name]["files"][params["path"]]
                    # Create a mock record with dict-like access
                    mock_record = MagicMock()
                    mock_record.__getitem__ = lambda self, key: {
                        "path": params["path"],
                        "project_name": project_name,
                        "language": file_data.get("language"),
                        "size": file_data.get("size"),
                        "hash": file_data.get("hash"),
                        "last_modified": file_data.get("last_modified"),
                        "chunk_count": 1,
                    }[key]
                    mock_record.get = lambda key, default=None: {
                        "path": params["path"],
                        "project_name": project_name,
                        "language": file_data.get("language"),
                        "size": file_data.get("size"),
                        "hash": file_data.get("hash"),
                        "last_modified": file_data.get("last_modified"),
                        "chunk_count": 1,
                    }.get(key, default)
                    return MagicMock(records=[mock_record])
                return MagicMock(records=[])

            elif "MATCH (c:CodeChunk" in query and "embedding" in query.lower():
                # Return search results with project context
                if project_name:
                    chunks = driver._storage[project_name]["chunks"]
                    results = []
                    for chunk in chunks[:5]:
                        # Create a mock record with dict-like access
                        mock_record = MagicMock()
                        record_data = {
                            "project_name": project_name,
                            "file_path": chunk["file_path"],
                            "chunk_content": chunk["content"],
                            "line_number": chunk["start_line"],
                            "similarity": 0.85,
                        }
                        mock_record.__getitem__ = lambda self, key, data=record_data: data[key]
                        mock_record.get = lambda key, default=None, data=record_data: data.get(key, default)
                        results.append(mock_record)
                    return MagicMock(records=results)
                return MagicMock(records=[])

            elif "count(DISTINCT f)" in query:
                # Return repository stats
                if project_name:
                    files = driver._storage[project_name]["files"]
                    chunks = driver._storage[project_name]["chunks"]
                    mock_record = MagicMock()
                    record_data = {
                        "total_files": len(files),
                        "total_chunks": len(chunks),
                        "total_size": sum(f.get("size", 0) for f in files.values()),
                        "languages": ["python"],
                        "project_name": project_name,
                    }
                    mock_record.__getitem__ = lambda self, key, data=record_data: data[key]
                    mock_record.get = lambda key, default=None, data=record_data: data.get(key, default)
                    return MagicMock(records=[mock_record])
                return MagicMock(records=[])

            return MagicMock(records=[])

        driver.execute_query.side_effect = execute_query_with_validation
        return driver

    async def test_project_name_required_for_initialization(self):
        """Test that Neo4jRAG requires project_name for initialization."""
        mock_driver = AsyncMock()

        # Should work with project_name
        rag = Neo4jRAG(
            neo4j_driver=mock_driver,
            project_name="test_project",
            embeddings=MockEmbeddingsProvider(),
        )
        assert rag.project_name == "test_project"

        # Project name should be stored and used
        assert hasattr(rag, "project_name")
        assert rag.project_name is not None
        assert isinstance(rag.project_name, str)

    async def test_all_operations_require_project_name(self, mock_driver_with_validation):
        """Test that all Neo4j operations include project_name parameter."""
        rag = Neo4jRAG(
            neo4j_driver=mock_driver_with_validation,
            project_name="validation_test",
            embeddings=MockEmbeddingsProvider(),
        )
        await rag.initialize()

        # Test index_file
        file = CodeFile(
            project_name="validation_test",
            path=Path("/test.py"),
            content="test content",
            language="python",
            size=12,
            last_modified=datetime.now(),
        )

        # Should succeed with proper project_name
        await rag.index_file(file)

        # Test search operations
        await rag.search_semantic("test query")
        await rag.search_by_pattern("test.*pattern")

        # Test metadata operations
        await rag.get_file_metadata(Path("/test.py"))
        await rag.get_repository_stats()

        # Test delete operation
        await rag.delete_file(Path("/test.py"))

        # All operations should have succeeded without raising ValueError
        # The mock driver validates that project_name was present in all queries

    async def test_project_name_stored_in_neo4j(self, mock_driver_with_validation):
        """Test that project_name is persisted in Neo4j nodes."""
        project_name = "persistence_test"
        rag = Neo4jRAG(
            neo4j_driver=mock_driver_with_validation,
            project_name=project_name,
            embeddings=MockEmbeddingsProvider(),
        )
        await rag.initialize()

        # Index a file
        file = CodeFile(
            project_name=project_name,
            path=Path("/persist/test.py"),
            content="def persist(): pass",
            language="python",
            size=20,
            last_modified=datetime.now(),
        )

        await rag.index_file(file)

        # Check that data was stored with project_name
        stored_files = mock_driver_with_validation._storage[project_name]["files"]
        assert "/persist/test.py" in stored_files
        assert stored_files["/persist/test.py"]["project_name"] == project_name

        stored_chunks = mock_driver_with_validation._storage[project_name]["chunks"]
        assert len(stored_chunks) > 0
        for chunk in stored_chunks:
            assert chunk["project_name"] == project_name

    async def test_search_results_include_project_context(self, mock_driver_with_validation):
        """Test that search results include project_name field."""
        project_name = "search_context_test"
        rag = Neo4jRAG(
            neo4j_driver=mock_driver_with_validation,
            project_name=project_name,
            embeddings=MockEmbeddingsProvider(),
        )
        await rag.initialize()

        # Index some content
        file = CodeFile(
            project_name=project_name,
            path=Path("/search/test.py"),
            content="def search_function(): return 'found'",
            language="python",
            size=38,
            last_modified=datetime.now(),
        )

        await rag.index_file(file)

        # Perform semantic search
        results = await rag.search_semantic("search function")

        # Verify results include project_name
        assert len(results) > 0
        for result in results:
            assert isinstance(result, SearchResult)
            assert result.project_name == project_name
            assert hasattr(result, "file_path")
            assert hasattr(result, "content")
            assert hasattr(result, "line_number")
            assert hasattr(result, "similarity")

        # Perform pattern search
        pattern_results = await rag.search_by_pattern("search")

        for result in pattern_results:
            assert isinstance(result, SearchResult)
            assert result.project_name == project_name

    async def test_dataclass_project_name_fields(self):
        """Test that all dataclasses have project_name field."""
        # Test CodeFile dataclass
        file = CodeFile(
            project_name="dataclass_test",
            path=Path("/test.py"),
            content="test",
            language="python",
            size=4,
            last_modified=datetime.now(),
        )
        assert file.project_name == "dataclass_test"

        # Test CodeChunk dataclass
        chunk = CodeChunk(
            project_name="dataclass_test",
            file_path=Path("/test.py"),
            content="chunk content",
            start_line=1,
            end_line=10,
            embedding=[0.1, 0.2],
        )
        assert chunk.project_name == "dataclass_test"

        # Test SearchResult dataclass
        result = SearchResult(
            project_name="dataclass_test",
            file_path=Path("/test.py"),
            content="result content",
            line_number=5,
            similarity=0.95,
        )
        assert result.project_name == "dataclass_test"

    async def test_repository_monitor_project_context(self):
        """Test that RepositoryMonitor maintains project context."""
        mock_driver = AsyncMock()
        project_name = "monitor_test"

        monitor = RepositoryMonitor(
            repo_path=Path("/test/repo"),
            project_name=project_name,
            neo4j_driver=mock_driver,
            file_patterns=["*.py"],
        )

        assert monitor.project_name == project_name

        # The monitor should pass project_name to RAG operations
        assert hasattr(monitor, "project_name")
        assert monitor.project_name == project_name

    async def test_mcp_server_project_context(self):
        """Test that MCP server tools maintain project context."""
        mock_monitor = AsyncMock()
        mock_monitor.repo_path = Path("/test/repo")
        mock_monitor.scan_repository = AsyncMock(return_value=[])
        mock_monitor.start = AsyncMock()

        mock_rag = AsyncMock()
        mock_rag.index_file = AsyncMock()
        mock_rag.search_semantic = AsyncMock(return_value=[])
        mock_rag.search_by_pattern = AsyncMock(return_value=[])
        mock_rag.get_repository_stats = AsyncMock(
            return_value={
                "project_name": "server_test",
                "total_files": 0,
                "total_chunks": 0,
                "total_size": 0,
                "languages": [],
            }
        )

        project_name = "server_test"

        # Create MCP server with project context
        server = create_mcp_server(
            repository_monitor=mock_monitor, neo4j_rag=mock_rag, project_name=project_name
        )

        assert server is not None

        # The server should have tools that respect project context
        # We can't directly test the tools here, but the server creation
        # with project_name should ensure tools use it

    async def test_project_name_consistency_enforcement(self, mock_driver_with_validation):
        """Test that project_name is enforced consistently."""
        project_name = "consistency_test"
        rag = Neo4jRAG(
            neo4j_driver=mock_driver_with_validation,
            project_name=project_name,
            embeddings=MockEmbeddingsProvider(),
        )
        await rag.initialize()

        # Try to index a file with wrong project_name
        wrong_file = CodeFile(
            project_name="wrong_project",  # Different project name
            path=Path("/test.py"),
            content="test",
            language="python",
            size=4,
            last_modified=datetime.now(),
        )

        # The RAG should correct the project_name
        await rag.index_file(wrong_file)

        # Check that the file was indexed with the correct project_name
        calls = mock_driver_with_validation.execute_query.call_args_list
        for call in calls:
            if call[0] and len(call[0]) > 1:
                params = call[0][1]
                if "project_name" in params:
                    assert params["project_name"] == project_name  # Should use RAG's project_name

    async def test_file_metadata_includes_project(self, mock_driver_with_validation):
        """Test that file metadata operations include project context."""
        project_name = "metadata_test"
        rag = Neo4jRAG(
            neo4j_driver=mock_driver_with_validation,
            project_name=project_name,
            embeddings=MockEmbeddingsProvider(),
        )
        await rag.initialize()

        # Index a file
        file_path = Path("/metadata/test.py")
        file = CodeFile(
            project_name=project_name,
            path=file_path,
            content="metadata test content",
            language="python",
            size=21,
            last_modified=datetime.now(),
        )

        await rag.index_file(file)

        # Mock the metadata response
        mock_driver_with_validation.execute_query.return_value = MagicMock(
            records=[
                {
                    "path": str(file_path),
                    "language": "python",
                    "size": 21,
                    "last_modified": "2024-01-01T10:00:00",
                    "hash": "abc123",
                    "project_name": project_name,
                    "chunk_count": 1,
                }
            ]
        )

        # Get file metadata
        metadata = await rag.get_file_metadata(file_path)

        assert metadata is not None
        assert metadata["project_name"] == project_name

        # Verify the query included project_name
        last_call = mock_driver_with_validation.execute_query.call_args
        assert last_call[0][1]["project_name"] == project_name

    async def test_update_file_maintains_project_context(self, mock_driver_with_validation):
        """Test that file updates maintain project context."""
        project_name = "update_test"
        rag = Neo4jRAG(
            neo4j_driver=mock_driver_with_validation,
            project_name=project_name,
            embeddings=MockEmbeddingsProvider(),
        )
        await rag.initialize()

        # Index initial file
        file_v1 = CodeFile(
            project_name=project_name,
            path=Path("/update/test.py"),
            content="# Version 1",
            language="python",
            size=11,
            last_modified=datetime.now(),
        )

        await rag.index_file(file_v1)

        # Update the file
        file_v2 = CodeFile(
            project_name=project_name,
            path=Path("/update/test.py"),
            content="# Version 2 - Updated",
            language="python",
            size=21,
            last_modified=datetime.now(),
        )

        # Mock that the file exists with different hash
        mock_driver_with_validation.execute_query.return_value = MagicMock(
            records=[{"hash": "old_hash"}]
        )

        await rag.update_file(file_v2)

        # Verify all update operations included project_name
        calls = mock_driver_with_validation.execute_query.call_args_list
        for call in calls:
            if call[0] and len(call[0]) > 1:
                params = call[0][1]
                if "project_name" in params:
                    assert params["project_name"] == project_name

    async def test_project_name_in_stats(self, mock_driver_with_validation):
        """Test that repository stats include project context."""
        project_name = "stats_test"
        rag = Neo4jRAG(
            neo4j_driver=mock_driver_with_validation,
            project_name=project_name,
            embeddings=MockEmbeddingsProvider(),
        )
        await rag.initialize()

        # Mock stats response
        mock_driver_with_validation.execute_query.return_value = MagicMock(
            records=[
                {
                    "total_files": 10,
                    "total_chunks": 50,
                    "total_size": 100000,
                    "languages": ["python", "javascript"],
                    "project_name": project_name,
                }
            ]
        )

        stats = await rag.get_repository_stats()

        assert stats["project_name"] == project_name
        assert stats["total_files"] == 10
        assert stats["total_chunks"] == 50

        # Verify the query included project_name filter
        last_call = mock_driver_with_validation.execute_query.call_args
        assert last_call[0][1]["project_name"] == project_name
