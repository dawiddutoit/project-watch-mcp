"""Test suite for MCP server functionality."""

from unittest.mock import AsyncMock

import pytest

from src.project_watch_mcp.server import create_mcp_server


@pytest.fixture
def mock_repository_monitor():
    """Mock RepositoryMonitor for testing."""
    monitor = AsyncMock()
    monitor.scan_repository = AsyncMock(return_value=[])
    monitor.start = AsyncMock()
    monitor.stop = AsyncMock()
    monitor.is_running = False
    return monitor


@pytest.fixture
def mock_neo4j_rag():
    """Mock Neo4jRAG for testing."""
    rag = AsyncMock()
    rag.initialize = AsyncMock()
    rag.search_semantic = AsyncMock(return_value=[])
    rag.search_by_pattern = AsyncMock(return_value=[])
    rag.get_repository_stats = AsyncMock(
        return_value={"total_files": 0, "total_chunks": 0, "total_size": 0, "languages": []}
    )
    return rag


@pytest.fixture
def mcp_server(mock_repository_monitor, mock_neo4j_rag):
    """Create MCP server for testing."""
    return create_mcp_server(
        repository_monitor=mock_repository_monitor,
        neo4j_rag=mock_neo4j_rag,
        project_name="test_project",
    )


class TestMCPServer:
    """Test suite for MCP server."""

    def test_server_creation(self, mcp_server):
        """Test that MCP server is created correctly."""
        assert mcp_server is not None
        assert mcp_server.name == "project-watch-mcp"

    async def test_has_initialize_repository_tool(self, mcp_server):
        """Test that initialize_repository tool is registered."""
        # Get tools from the MCP server - returns list of tool names
        tools = await mcp_server.get_tools()

        # Check that our tool is registered
        assert "initialize_repository" in tools

    async def test_has_search_code_tool(self, mcp_server):
        """Test that search_code tool is registered."""
        tools = await mcp_server.get_tools()
        assert "search_code" in tools

    async def test_has_get_repository_stats_tool(self, mcp_server):
        """Test that get_repository_stats tool is registered."""
        tools = await mcp_server.get_tools()
        assert "get_repository_stats" in tools

    async def test_has_get_file_info_tool(self, mcp_server):
        """Test that get_file_info tool is registered."""
        tools = await mcp_server.get_tools()
        assert "get_file_info" in tools

    async def test_has_refresh_file_tool(self, mcp_server):
        """Test that refresh_file tool is registered."""
        tools = await mcp_server.get_tools()
        assert "refresh_file" in tools

    async def test_has_monitoring_status_tool(self, mcp_server):
        """Test that monitoring_status tool is registered."""
        tools = await mcp_server.get_tools()
        assert "monitoring_status" in tools

    async def test_all_expected_tools_registered(self, mcp_server):
        """Test that all expected tools are registered."""
        tools = await mcp_server.get_tools()

        expected_tools = [
            "initialize_repository",
            "search_code",
            "get_repository_stats",
            "get_file_info",
            "refresh_file",
            "monitoring_status",
        ]

        for tool_name in expected_tools:
            assert tool_name in tools, f"Tool {tool_name} not registered"
