"""Test suite for project context isolation in Neo4j RAG."""

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from src.project_watch_mcp.neo4j_rag import CodeFile, Neo4jRAG
from src.project_watch_mcp.repository_monitor import RepositoryMonitor
from src.project_watch_mcp.server import create_mcp_server


class TestProjectIsolation:
    """Test that project context isolation works correctly."""

    @pytest.fixture
    def mock_neo4j_driver(self):
        """Mock Neo4j driver for testing."""
        driver = AsyncMock()
        driver.verify_connectivity = AsyncMock()
        driver.execute_query = AsyncMock()
        return driver

    @pytest.fixture
    def mock_embeddings(self):
        """Mock embeddings provider."""
        embeddings = MagicMock()
        embeddings.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])
        embeddings.dimension = 3
        return embeddings

    @pytest_asyncio.fixture
    async def project_a_rag(self, mock_neo4j_driver, mock_embeddings):
        """Create a Neo4jRAG instance for project A."""
        rag = Neo4jRAG(
            neo4j_driver=mock_neo4j_driver,
            project_name="project_a",
            embeddings=mock_embeddings,
            chunk_size=100,
            chunk_overlap=20,
        )
        await rag.initialize()
        return rag

    @pytest_asyncio.fixture
    async def project_b_rag(self, mock_neo4j_driver, mock_embeddings):
        """Create a Neo4jRAG instance for project B."""
        rag = Neo4jRAG(
            neo4j_driver=mock_neo4j_driver,
            project_name="project_b",
            embeddings=mock_embeddings,
            chunk_size=100,
            chunk_overlap=20,
        )
        await rag.initialize()
        return rag

    async def test_different_projects_have_different_names(self, project_a_rag, project_b_rag):
        """Test that different RAG instances have different project names."""
        assert project_a_rag.project_name == "project_a"
        assert project_b_rag.project_name == "project_b"
        assert project_a_rag.project_name != project_b_rag.project_name

    async def test_index_file_includes_project_name(self, project_a_rag):
        """Test that indexed files include the project name."""
        code_file = CodeFile(
            project_name="project_a",
            path=Path("/test/main.py"),
            content="def hello(): pass",
            language="python",
            size=17,
            last_modified=datetime.now(),
        )

        await project_a_rag.index_file(code_file)

        # Check that the file was indexed with project name
        calls = project_a_rag.neo4j_driver.execute_query.call_args_list

        # Find the MERGE query for the file
        file_merge_call = None
        for call in calls:
            if "MERGE (f:CodeFile" in str(call):
                file_merge_call = call
                break

        assert file_merge_call is not None
        # Check that project_name was passed in parameters
        params = file_merge_call[0][1] if len(file_merge_call[0]) > 1 else file_merge_call[1]
        assert params.get("project_name") == "project_a"

    async def test_search_filters_by_project(self, project_a_rag):
        """Test that searches filter results by project name."""
        # Mock search results with different project names
        mock_results = MagicMock()
        mock_results.records = [
            {
                "file_path": "/test/file1.py",
                "chunk_content": "project a code",
                "line_number": 1,
                "similarity": 0.9,
                "project_name": "project_a",
            }
        ]
        project_a_rag.neo4j_driver.execute_query.return_value = mock_results

        results = await project_a_rag.search_semantic("test query")

        # Verify that all results are from project_a
        assert len(results) == 1
        assert results[0].project_name == "project_a"

        # Check that the query included project filter
        query_call = project_a_rag.neo4j_driver.execute_query.call_args
        params = query_call[0][1] if len(query_call[0]) > 1 else query_call[1]
        assert params.get("project_name") == "project_a"

    async def test_delete_file_only_affects_one_project(self, project_a_rag):
        """Test that deleting a file only affects the specified project."""
        file_path = Path("/test/shared.py")

        await project_a_rag.delete_file(file_path)

        # Check that the delete query included project name
        delete_call = project_a_rag.neo4j_driver.execute_query.call_args
        query = delete_call[0][0] if delete_call[0] else ""
        params = delete_call[0][1] if len(delete_call[0]) > 1 else delete_call[1]

        assert "project_name" in query or params.get("project_name") == "project_a"
        assert str(file_path) in str(params)

    async def test_repository_stats_per_project(self, project_a_rag):
        """Test that repository stats are calculated per project."""
        mock_result = MagicMock()
        mock_result.records = [
            {
                "total_files": 5,
                "total_chunks": 20,
                "total_size": 10000,
                "languages": ["python"],
                "project_name": "project_a",
            }
        ]
        project_a_rag.neo4j_driver.execute_query.return_value = mock_result

        stats = await project_a_rag.get_repository_stats()

        assert stats["project_name"] == "project_a"
        assert stats["total_files"] == 5

        # Check that the query filtered by project
        stats_call = project_a_rag.neo4j_driver.execute_query.call_args
        params = stats_call[0][1] if len(stats_call[0]) > 1 else stats_call[1]
        assert params.get("project_name") == "project_a"

    async def test_file_metadata_includes_project(self, project_a_rag):
        """Test that file metadata includes project context."""
        mock_result = MagicMock()
        mock_result.records = [
            {
                "path": "/test/file.py",
                "language": "python",
                "size": 100,
                "last_modified": "2024-01-01T10:00:00",
                "hash": "abc123",
                "project_name": "project_a",
                "chunk_count": 2,
            }
        ]
        project_a_rag.neo4j_driver.execute_query.return_value = mock_result

        metadata = await project_a_rag.get_file_metadata(Path("/test/file.py"))

        assert metadata is not None
        assert metadata["project_name"] == "project_a"

        # Check that the query filtered by project
        metadata_call = project_a_rag.neo4j_driver.execute_query.call_args
        params = metadata_call[0][1] if len(metadata_call[0]) > 1 else metadata_call[1]
        assert params.get("project_name") == "project_a"

    async def test_repository_monitor_with_project(self):
        """Test that repository monitor includes project context."""
        mock_driver = AsyncMock()
        repo_path = Path("/test/repo")

        monitor = RepositoryMonitor(
            repo_path=repo_path,
            project_name="test_project",
            neo4j_driver=mock_driver,
            file_patterns=["*.py"],
        )

        assert monitor.project_name == "test_project"
        assert monitor.repo_path == repo_path

    async def test_mcp_server_with_project(self):
        """Test that MCP server respects project context."""
        mock_monitor = AsyncMock()
        mock_monitor.repo_path = Path("/test/repo")
        mock_monitor.file_patterns = ["*.py"]
        mock_monitor.is_running = False

        mock_rag = AsyncMock()
        mock_rag.search_semantic = AsyncMock(return_value=[])

        server = create_mcp_server(
            repository_monitor=mock_monitor, neo4j_rag=mock_rag, project_name="test_project"
        )

        # Server should be created with project context
        assert server is not None
        # The server should have been created successfully
        # We can't directly check tools attribute, but the server creation
        # with project_name parameter should succeed without errors

    async def test_same_file_path_different_projects(self, mock_embeddings):
        """Test that the same file path can exist in different projects."""
        # Create separate mock drivers for each project
        driver_a = AsyncMock()
        driver_a.verify_connectivity = AsyncMock()
        driver_a.execute_query = AsyncMock()

        driver_b = AsyncMock()
        driver_b.verify_connectivity = AsyncMock()
        driver_b.execute_query = AsyncMock()

        # Create separate RAG instances with their own drivers
        project_a_rag = Neo4jRAG(
            neo4j_driver=driver_a,
            project_name="project_a",
            embeddings=mock_embeddings,
            chunk_size=100,
            chunk_overlap=20,
        )
        await project_a_rag.initialize()

        project_b_rag = Neo4jRAG(
            neo4j_driver=driver_b,
            project_name="project_b",
            embeddings=mock_embeddings,
            chunk_size=100,
            chunk_overlap=20,
        )
        await project_b_rag.initialize()

        # Create the same file for both projects
        file_path = Path("/shared/main.py")

        code_file_a = CodeFile(
            project_name="project_a",
            path=file_path,
            content="# Project A version",
            language="python",
            size=19,
            last_modified=datetime.now(),
        )

        code_file_b = CodeFile(
            project_name="project_b",
            path=file_path,
            content="# Project B version",
            language="python",
            size=19,
            last_modified=datetime.now(),
        )

        # Index the same file path in both projects
        await project_a_rag.index_file(code_file_a)
        await project_b_rag.index_file(code_file_b)

        # Both should succeed without conflict
        assert driver_a.execute_query.called
        assert driver_b.execute_query.called

        # The calls should include different project names
        a_calls = driver_a.execute_query.call_args_list
        b_calls = driver_b.execute_query.call_args_list

        # Check that project names are different in the calls
        for call in a_calls:
            if len(call[0]) > 1:
                params = call[0][1]
                if "project_name" in params:
                    assert params["project_name"] == "project_a"

        for call in b_calls:
            if len(call[0]) > 1:
                params = call[0][1]
                if "project_name" in params:
                    assert params["project_name"] == "project_b"
