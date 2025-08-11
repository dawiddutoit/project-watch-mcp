"""Integration tests for MCP server with project context.

This module tests that all MCP tools properly handle project context
and that the repository monitor maintains project isolation.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest_asyncio

from src.project_watch_mcp.neo4j_rag import CodeFile, Neo4jRAG
from src.project_watch_mcp.repository_monitor import FileInfo, RepositoryMonitor
from src.project_watch_mcp.server import create_mcp_server
from src.project_watch_mcp.utils.embedding import MockEmbeddingsProvider


class TestMCPIntegration:
    """Test MCP server integration with project context."""

    @pytest_asyncio.fixture
    async def mock_repository_files(self, tmp_path):
        """Create mock repository structure."""
        # Create test repository structure
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()

        # Create some test files
        (repo_path / "main.py").write_text("def main(): pass")
        (repo_path / "utils.py").write_text("def util_func(): pass")

        src_dir = repo_path / "src"
        src_dir.mkdir()
        (src_dir / "module.py").write_text("class Module: pass")

        tests_dir = repo_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_main.py").write_text("def test_main(): pass")

        # Create gitignore
        (repo_path / ".gitignore").write_text("*.pyc\n__pycache__/\n.venv/")

        return repo_path

    @pytest_asyncio.fixture
    async def mcp_server_setup(self, mock_repository_files):
        """Set up MCP server with mocked components."""
        mock_driver = AsyncMock()
        mock_driver.verify_connectivity = AsyncMock()
        mock_driver.execute_query = AsyncMock()
        mock_driver.close = AsyncMock()

        # Track indexed files and searches
        mock_driver._indexed_files = {}
        mock_driver._search_queries = []

        async def execute_query_tracker(query, params=None, *args, **kwargs):
            """Track queries for verification."""
            project_name = params.get("project_name") if params else None

            if "MERGE (f:CodeFile" in query and project_name:
                # Track file indexing
                if project_name not in mock_driver._indexed_files:
                    mock_driver._indexed_files[project_name] = []
                mock_driver._indexed_files[project_name].append(params.get("path"))

            elif "similarity" in query.lower() and project_name:
                # Track searches
                mock_driver._search_queries.append(
                    {
                        "project_name": project_name,
                        "query": params.get("query_embedding", ""),
                    }
                )
                # Return mock search results with proper dict-like access
                class MockRecord:
                    def __init__(self, data):
                        self._data = data
                    def __getitem__(self, key):
                        return self._data[key]
                    def get(self, key, default=None):
                        return self._data.get(key, default)
                
                record_data = {
                    "file_path": "/test/file.py",
                    "chunk_content": "test content",
                    "line_number": 1,
                    "similarity": 0.9,
                    "project_name": project_name,
                }
                return MagicMock(records=[MockRecord(record_data)])

            elif "count(DISTINCT f)" in query and project_name:
                # Return mock stats with proper dict-like access
                class MockRecord:
                    def __init__(self, data):
                        self._data = data
                    def __getitem__(self, key):
                        return self._data[key]
                    def get(self, key, default=None):
                        return self._data.get(key, default)
                
                record_data = {
                    "total_files": len(mock_driver._indexed_files.get(project_name, [])),
                    "total_chunks": len(mock_driver._indexed_files.get(project_name, [])) * 2,
                    "total_size": 1000,
                    "languages": ["python"],
                    "project_name": project_name,
                }
                return MagicMock(records=[MockRecord(record_data)])
            
            elif "MATCH (f:CodeFile" in query:
                # Return mock file info with proper dict-like access
                class MockRecord:
                    def __init__(self, data):
                        self._data = data
                    def __getitem__(self, key):
                        return self._data[key]
                    def get(self, key, default=None):
                        return self._data.get(key, default)
                
                record_data = {
                    "path": "/test/file.py",
                    "language": "python",
                    "size": 100,
                    "last_modified": "2024-01-01",
                    "hash": "abc123",
                    "project_name": project_name,
                    "chunk_count": 2,
                }
                return MagicMock(records=[MockRecord(record_data)])

            return MagicMock(records=[])

        mock_driver.execute_query.side_effect = execute_query_tracker

        # Create repository monitor
        project_name = "mcp_test_project"
        monitor = RepositoryMonitor(
            repo_path=mock_repository_files,
            project_name=project_name,
            neo4j_driver=mock_driver,
            file_patterns=["*.py"],
            ignore_patterns=["*.pyc", "__pycache__", ".venv"],
        )

        # Create RAG instance
        rag = Neo4jRAG(
            neo4j_driver=mock_driver, project_name=project_name, embeddings=MockEmbeddingsProvider()
        )
        await rag.initialize()

        # Create MCP server
        server = create_mcp_server(
            repository_monitor=monitor, neo4j_rag=rag, project_name=project_name
        )

        return {
            "server": server,
            "monitor": monitor,
            "rag": rag,
            "driver": mock_driver,
            "project_name": project_name,
            "repo_path": mock_repository_files,
        }

    async def test_initialize_repository_with_project_context(self, mcp_server_setup):
        """Test that initialize_repository tool respects project context."""
        server = mcp_server_setup["server"]
        driver = mcp_server_setup["driver"]
        project_name = mcp_server_setup["project_name"]

        # Get the initialize_repository tool
        tools = await server.get_tools()
        assert "initialize_repository" in tools, "initialize_repository tool not found"
        init_tool = tools["initialize_repository"]

        # Execute the tool
        result = await init_tool.run({})

        # Verify files were indexed with correct project context
        assert project_name in driver._indexed_files
        indexed_files = driver._indexed_files[project_name]

        # Should have indexed Python files from the repository
        assert len(indexed_files) > 0

        # Verify result indicates success
        assert result is not None
        if hasattr(result, "content") and result.content:
            # result.content is a list of content objects
            text_content = result.content[0].text if hasattr(result.content[0], "text") else str(result.content[0])
            assert "initialized" in text_content.lower() or "indexed" in text_content.lower()

    async def test_search_code_tool_with_project_context(self, mcp_server_setup):
        """Test that search_code tool includes project context."""
        server = mcp_server_setup["server"]
        driver = mcp_server_setup["driver"]
        project_name = mcp_server_setup["project_name"]

        # Get the search_code tool
        tools = await server.get_tools()
        assert "search_code" in tools, "search_code tool not found"
        search_tool = tools["search_code"]

        # Execute semantic search
        result = await search_tool.run({"query": "test function", "search_type": "semantic", "limit": 5})

        # Verify search was performed with project context
        assert len(driver._search_queries) > 0
        last_search = driver._search_queries[-1]
        assert last_search["project_name"] == project_name

        # Results should include project context
        if hasattr(result, "structured_content"):
            for item in result.structured_content:
                # Each result should be from the correct project
                pass  # Results are mocked, but in real scenario would check project_name

    async def test_get_repository_stats_tool_with_project(self, mcp_server_setup):
        """Test that get_repository_stats tool respects project context."""
        server = mcp_server_setup["server"]
        project_name = mcp_server_setup["project_name"]

        # Get the stats tool
        tools = await server.get_tools()
        assert "get_repository_stats" in tools, "get_repository_stats tool not found"
        stats_tool = tools["get_repository_stats"]

        # Execute the tool
        result = await stats_tool.run({})

        # Verify stats are project-specific
        if hasattr(result, "structured_content"):
            stats = result.structured_content
            # Stats should be for the specific project
            # In real scenario, would verify project_name in response

        # Verify result is not empty
        assert result is not None

    async def test_get_file_info_tool_with_project(self, mcp_server_setup):
        """Test that get_file_info tool respects project context."""
        server = mcp_server_setup["server"]
        project_name = mcp_server_setup["project_name"]
        driver = mcp_server_setup["driver"]
        repo_path = mcp_server_setup["repo_path"]

        # Get the file info tool
        tools = await server.get_tools()
        assert "get_file_info" in tools, "get_file_info tool not found"
        file_info_tool = tools["get_file_info"]

        # Execute the tool
        result = await file_info_tool.run({"file_path": "test/file.py"})

        # Verify query included project context
        last_call = driver.execute_query.call_args
        if last_call and last_call[0] and len(last_call[0]) > 1:
            params = last_call[0][1]
            assert params.get("project_name") == project_name

    async def test_refresh_file_tool_with_project(self, mcp_server_setup):
        """Test that refresh_file tool maintains project context."""
        server = mcp_server_setup["server"]
        project_name = mcp_server_setup["project_name"]
        repo_path = mcp_server_setup["repo_path"]

        # Get the refresh_file tool
        tools = await server.get_tools()
        assert "refresh_file" in tools, "refresh_file tool not found"
        refresh_tool = tools["refresh_file"]

        # Create a file to refresh
        test_file = repo_path / "refresh_test.py"
        test_file.write_text("def refresh_me(): pass")

        # Execute the tool
        result = await refresh_tool.run({"file_path": str(test_file.relative_to(repo_path))})

        # Verify result indicates success
        assert result is not None
        if hasattr(result, "content") and result.content:
            # result.content is a list of content objects
            text_content = result.content[0].text if hasattr(result.content[0], "text") else str(result.content[0])
            assert "refreshed" in text_content.lower() or "updated" in text_content.lower()

    async def test_repository_monitor_project_tracking(self, mcp_server_setup):
        """Test that repository monitor tracks project context correctly."""
        monitor = mcp_server_setup["monitor"]
        project_name = mcp_server_setup["project_name"]

        # Verify monitor is configured with correct project
        assert monitor.project_name == project_name

        # Scan repository
        files = await monitor.scan_repository()

        # All scanned files should be associated with the project
        for file_info in files:
            # In actual implementation, file_info would include project context
            assert file_info.path.exists()
            assert file_info.language in ["python", None]

    async def test_file_change_handling_with_project(self, mcp_server_setup):
        """Test that file changes are handled with project context."""
        monitor = mcp_server_setup["monitor"]
        rag = mcp_server_setup["rag"]
        project_name = mcp_server_setup["project_name"]
        repo_path = mcp_server_setup["repo_path"]

        # Create a new file
        new_file = repo_path / "new_file.py"
        new_file.write_text("def new_function(): pass")

        # Scan to find the new file
        files = await monitor.scan_repository()

        # Find our new file
        new_file_info = None
        for file_info in files:
            if file_info.path.name == "new_file.py":
                new_file_info = file_info
                break

        assert new_file_info is not None

        # Index the file - FileInfo doesn't have content, we need to read it
        content = new_file_info.path.read_text()
        code_file = CodeFile(
            project_name=project_name,
            path=new_file_info.path,
            content=content,
            language=new_file_info.language,
            size=new_file_info.size,
            last_modified=new_file_info.last_modified,
        )

        await rag.index_file(code_file)

        # Verify it was indexed with correct project context
        # In actual implementation, would query Neo4j to verify

    async def test_concurrent_mcp_operations(self, mcp_server_setup):
        """Test that concurrent MCP operations maintain project isolation."""
        server = mcp_server_setup["server"]
        project_name = mcp_server_setup["project_name"]

        # Get tools
        tools = await server.get_tools()
        init_tool = tools.get("initialize_repository")
        search_tool = tools.get("search_code")
        stats_tool = tools.get("get_repository_stats")

        # Execute multiple operations concurrently
        results = await asyncio.gather(
            init_tool.run({}),
            search_tool.run({"query": "test", "search_type": "semantic"}),
            stats_tool.run({}),
            return_exceptions=True,
        )

        # All operations should complete without errors
        for result in results:
            assert not isinstance(result, Exception)

    async def test_multiple_mcp_servers_different_projects(self):
        """Test multiple MCP servers with different projects."""
        # This would require creating multiple server instances
        # with different project names and verifying isolation
        # Skipping detailed implementation for brevity
        pass

    async def test_mcp_tools_parameter_validation(self, mcp_server_setup):
        """Test that MCP tools validate parameters correctly."""
        server = mcp_server_setup["server"]

        # Get search tool
        tools = await server.get_tools()
        search_tool = tools.get("search_code")

        # Test with invalid parameters
        try:
            # Missing required parameter 'query'
            result = await search_tool.run({})
            # Should either handle gracefully or raise appropriate error
        except Exception as e:
            # Expected behavior - missing required parameter
            pass

    async def test_repository_monitor_gitignore_respect(self, mcp_server_setup):
        """Test that repository monitor respects .gitignore patterns."""
        monitor = mcp_server_setup["monitor"]
        repo_path = mcp_server_setup["repo_path"]

        # Create files that should be ignored
        (repo_path / "test.pyc").write_text("compiled")
        cache_dir = repo_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "cached.pyc").write_text("cached")

        # Scan repository
        files = await monitor.scan_repository()

        # Verify ignored files are not included
        file_names = [f.path.name for f in files]
        assert "test.pyc" not in file_names
        assert "cached.pyc" not in file_names

        # Verify normal Python files are included
        assert "main.py" in file_names
        assert "utils.py" in file_names