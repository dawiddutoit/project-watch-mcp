"""Comprehensive tests for MCP server module covering all tools."""

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import time

import pytest
from fastmcp.exceptions import ToolError

from project_watch_mcp.neo4j_rag import SearchResult, CodeFile
from project_watch_mcp.repository_monitor import FileInfo
from project_watch_mcp.server import create_mcp_server
from project_watch_mcp.complexity_analysis import ComplexityResult, FunctionComplexity


@pytest.fixture
def mock_repository_monitor():
    """Create a mock repository monitor with comprehensive mocks."""
    monitor = AsyncMock()
    monitor.repo_path = Path("/test/repo")
    monitor.file_patterns = ["*.py", "*.js", "*.ts", "*.java", "*.kt"]
    monitor.is_running = False
    monitor.monitoring_since = datetime.now()
    
    # Create sample file info objects
    sample_files = [
        FileInfo(
            path=Path("/test/repo/main.py"),
            size=1024,
            last_modified=datetime.now(),
            language="python"
        ),
        FileInfo(
            path=Path("/test/repo/utils.js"),
            size=512,
            last_modified=datetime.now(),
            language="javascript"
        )
    ]
    
    monitor.scan_repository = AsyncMock(return_value=sample_files)
    monitor.start = AsyncMock()
    monitor.stop = AsyncMock()
    monitor.process_all_changes = AsyncMock(return_value=[])
    monitor.get_file_info = AsyncMock(return_value=sample_files[0])
    monitor.pending_changes = []
    monitor.recent_changes = []
    return monitor


@pytest.fixture
def mock_neo4j_rag():
    """Create a mock Neo4j RAG instance with comprehensive responses."""
    rag = AsyncMock()
    
    # Mock successful indexing
    rag.index_file = AsyncMock()
    
    # Mock semantic search results
    rag.search_semantic = AsyncMock(return_value=[
        SearchResult(
            project_name="test_project",
            file_path=Path("src/auth.py"),
            content="def authenticate(user, password):\n    # Check credentials",
            line_number=10,
            similarity=0.95,
            context="Authentication module"
        ),
        SearchResult(
            project_name="test_project",
            file_path=Path("src/jwt.py"),
            content="def validate_token(token):\n    # Validate JWT",
            line_number=25,
            similarity=0.85,
            context="JWT validation"
        )
    ])
    
    # Mock pattern search results
    rag.search_by_pattern = AsyncMock(return_value=[
        SearchResult(
            project_name="test_project",
            file_path=Path("src/utils.py"),
            content="# TODO: Implement error handling",
            line_number=15,
            similarity=1.0,
            context="Utility functions"
        )
    ])
    
    # Mock repository stats
    rag.get_repository_stats = AsyncMock(
        return_value={
            "total_files": 42,
            "total_chunks": 256,
            "total_size": 1048576,
            "total_lines": 5000,
            "languages": {
                "python": {"files": 25, "size": 512000, "percentage": 48.8},
                "javascript": {"files": 10, "size": 256000, "percentage": 24.4},
                "typescript": {"files": 7, "size": 280576, "percentage": 26.8}
            },
            "largest_files": [
                {"path": "src/main.py", "size": 10240, "lines": 300},
                {"path": "src/utils.js", "size": 8192, "lines": 250}
            ],
            "project_name": "test_project",
        }
    )
    
    # Mock file metadata
    sample_code_file = CodeFile(
        project_name="test_project",
        path=Path("src/main.py"),
        content="import os\n\nclass MainClass:\n    pass\n\n" + "def main():\n    print('Hello')\n" * 10,  # Make it 50 lines
        language="python",
        size=1024,
        last_modified=datetime.now(),
        filename="main.py",
        is_test=False,
        is_config=False
    )
    # Add imports and classes attributes
    sample_code_file.imports = ["os", "sys"]
    sample_code_file.classes = ["MainClass"]
    sample_code_file.functions = ["main"]
    sample_code_file.lines = 50
    rag.get_file_metadata = AsyncMock(return_value=sample_code_file)
    
    # Mock update and delete operations
    rag.update_file = AsyncMock(return_value={"chunks_before": 5, "chunks_after": 7})
    rag.delete_file = AsyncMock(return_value={"chunks_removed": 5})
    
    return rag


@pytest.fixture
def mcp_server(mock_repository_monitor, mock_neo4j_rag):
    """Create an MCP server with mocked dependencies."""
    return create_mcp_server(mock_repository_monitor, mock_neo4j_rag, project_name="test_project")


class TestMCPServer:
    """Comprehensive tests for MCP server and all its tools."""

    def test_server_creation(self, mcp_server):
        """Test that MCP server is created correctly."""
        assert mcp_server is not None
        assert mcp_server.name == "project-watch-mcp"

    @pytest.mark.asyncio
    async def test_all_tools_registered(self, mcp_server):
        """Test that all 8 expected tools are registered."""
        tools = await mcp_server.get_tools()
        
        # get_tools() returns a dict of tool_name -> FunctionTool
        if isinstance(tools, dict):
            tool_names = list(tools.keys())
        else:
            tool_names = tools
        
        expected_tools = [
            "initialize_repository",
            "search_code",
            "get_repository_stats",
            "get_file_info",
            "refresh_file",
            "delete_file",
            "analyze_complexity",
            "monitoring_status",
        ]
        
        for tool_name in expected_tools:
            assert tool_name in tool_names, f"Tool {tool_name} not registered"
            
        # Verify we have exactly 8 tools
        assert len([t for t in tool_names if t in expected_tools]) == 8

    @pytest.mark.asyncio
    async def test_initialize_repository_success(self, mcp_server, mock_repository_monitor, mock_neo4j_rag):
        """Test successful repository initialization."""
        # Execute the tool
        tool = await mcp_server.get_tool("initialize_repository")
        result = await tool.fn()
        
        # Verify the tool executed successfully
        assert result is not None
        assert "indexed" in result.structured_content
        assert "total" in result.structured_content
        assert result.structured_content["indexed"] == 2  # We have 2 sample files
        assert result.structured_content["total"] == 2
        
        # Verify mocks were called
        mock_repository_monitor.scan_repository.assert_called_once()
        assert mock_neo4j_rag.index_file.call_count == 2
        mock_repository_monitor.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_repository_partial_failure(self, mcp_server, mock_repository_monitor, mock_neo4j_rag):
        """Test repository initialization with some files failing to index."""
        # Make indexing fail for second file
        mock_neo4j_rag.index_file.side_effect = [None, Exception("Index failed")]
        
        tool = await mcp_server.get_tool("initialize_repository")
        result = await tool.fn()
        
        # Should still succeed but report skipped files
        assert result.structured_content["indexed"] == 1
        assert result.structured_content["total"] == 2
        assert len(result.structured_content["skipped"]) == 1

    @pytest.mark.asyncio
    async def test_initialize_repository_total_failure(self, mcp_server, mock_repository_monitor):
        """Test repository initialization failure."""
        mock_repository_monitor.scan_repository.side_effect = Exception("Scan failed")
        
        tool = await mcp_server.get_tool("initialize_repository")
        
        with pytest.raises(ToolError) as exc_info:
            await tool.fn()
        assert "Failed to initialize repository" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_code_semantic(self, mcp_server, mock_neo4j_rag):
        """Test semantic code search."""
        tool = await mcp_server.get_tool("search_code")
        result = await tool.fn(
            query="authentication logic",
            search_type="semantic",
            limit=5
        )
        
        assert result is not None
        assert "results" in result.structured_content
        assert len(result.structured_content["results"]) == 2
        assert result.structured_content["results"][0]["similarity"] == 0.95
        
        mock_neo4j_rag.search_semantic.assert_called_once_with(
            query="authentication logic",
            limit=5,
            language=None
        )

    @pytest.mark.asyncio
    async def test_search_code_pattern(self, mcp_server, mock_neo4j_rag):
        """Test pattern-based code search."""
        tool = await mcp_server.get_tool("search_code")
        result = await tool.fn(
            query="TODO",
            search_type="pattern",
            is_regex=False,
            limit=10
        )
        
        assert result is not None
        assert len(result.structured_content["results"]) == 1
        assert "TODO" in result.structured_content["results"][0]["content"]
        
        mock_neo4j_rag.search_by_pattern.assert_called_once_with(
            pattern="TODO",
            is_regex=False,
            limit=10,
            language=None
        )

    @pytest.mark.asyncio
    async def test_search_code_with_language_filter(self, mcp_server, mock_neo4j_rag):
        """Test code search with language filtering."""
        tool = await mcp_server.get_tool("search_code")
        result = await tool.fn(
            query="function",
            search_type="semantic",
            language="javascript"
        )
        
        mock_neo4j_rag.search_semantic.assert_called_with(
            query="function",
            limit=10,
            language="javascript"
        )

    @pytest.mark.asyncio
    async def test_search_code_limit_validation(self, mcp_server):
        """Test that search limit is capped at 100."""
        tool = await mcp_server.get_tool("search_code")
        result = await tool.fn(
            query="test",
            limit=200  # Over max
        )
        
        # Should be capped at 100
        assert result.structured_content["limit_applied"] <= 100

    @pytest.mark.asyncio
    async def test_get_repository_stats(self, mcp_server, mock_neo4j_rag):
        """Test getting repository statistics."""
        tool = await mcp_server.get_tool("get_repository_stats")
        result = await tool.fn()
        
        assert result is not None
        stats = result.structured_content
        assert stats["total_files"] == 42
        assert stats["total_chunks"] == 256
        assert stats["total_size"] == 1048576
        assert "python" in stats["languages"]
        assert len(stats["largest_files"]) == 2
        
        mock_neo4j_rag.get_repository_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_file_info_success(self, mcp_server, mock_neo4j_rag):
        """Test getting file information."""
        tool = await mcp_server.get_tool("get_file_info")
        result = await tool.fn(file_path="src/main.py")
        
        assert result is not None
        info = result.structured_content
        assert info["path"] == "src/main.py"
        assert info["language"] == "python"
        assert info["size"] == 1024
        assert info["lines"] == 50
        assert "os" in info["imports"]
        assert "MainClass" in info["classes"]
        
        mock_neo4j_rag.get_file_metadata.assert_called_once_with(Path("/test/repo/src/main.py"))

    @pytest.mark.asyncio
    async def test_get_file_info_not_found(self, mcp_server, mock_neo4j_rag):
        """Test getting info for non-existent file."""
        mock_neo4j_rag.get_file_metadata.return_value = None
        
        tool = await mcp_server.get_tool("get_file_info")
        
        result = await tool.fn(file_path="nonexistent.py")
        assert result.structured_content["error"] == "File not found in index"
        assert "not found in index" in result.content[0].text

    @pytest.mark.asyncio
    async def test_refresh_file_success(self, mcp_server, mock_repository_monitor, mock_neo4j_rag):
        """Test refreshing a file in the index."""
        tool = await mcp_server.get_tool("refresh_file")
        result = await tool.fn(file_path="src/main.py")
        
        assert result is not None
        assert result.structured_content["status"] == "success"
        assert result.structured_content["action"] == "updated"
        assert result.structured_content["chunks_before"] == 5
        assert result.structured_content["chunks_after"] == 7
        
        mock_repository_monitor.get_file_info.assert_called_once()
        mock_neo4j_rag.update_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_refresh_file_not_found(self, mcp_server, mock_repository_monitor):
        """Test refreshing a non-existent file."""
        mock_repository_monitor.get_file_info.return_value = None
        
        tool = await mcp_server.get_tool("refresh_file")
        
        with pytest.raises(ToolError) as exc_info:
            await tool.fn(file_path="nonexistent.py")
        assert "not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_delete_file_success(self, mcp_server, mock_neo4j_rag):
        """Test deleting a file from the index."""
        tool = await mcp_server.get_tool("delete_file")
        result = await tool.fn(file_path="src/old.py")
        
        assert result is not None
        assert result.structured_content["status"] == "success"
        assert result.structured_content["chunks_removed"] == 5
        
        mock_neo4j_rag.delete_file.assert_called_once_with(Path("/test/repo/src/old.py"))

    @pytest.mark.asyncio
    async def test_delete_file_not_found(self, mcp_server, mock_neo4j_rag):
        """Test deleting a non-indexed file."""
        mock_neo4j_rag.delete_file.return_value = {"chunks_removed": 0}
        
        tool = await mcp_server.get_tool("delete_file")
        result = await tool.fn(file_path="notindexed.py")
        
        assert result.structured_content["status"] == "warning"
        assert "not found in index" in result.structured_content["message"]

    @pytest.mark.asyncio
    @patch('project_watch_mcp.server.RADON_AVAILABLE', True)
    @patch('project_watch_mcp.server.cc_visit')
    @patch('project_watch_mcp.server.mi_visit')
    async def test_analyze_complexity_python(self, mock_mi, mock_cc, mcp_server):
        """Test analyzing Python file complexity."""
        # Mock radon responses
        mock_cc.return_value = [
            MagicMock(
                name="test_function",
                complexity=5,
                lineno=10,
                classname=None
            )
        ]
        mock_mi.return_value = 75.5
        
        # Create a temporary Python file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def test_function():\n    if True:\n        return 1\n    return 0")
            temp_path = f.name
        
        try:
            tool = await mcp_server.get_tool("analyze_complexity")
            result = await tool.fn(file_path=temp_path)
            
            assert result is not None
            assert "summary" in result.structured_content
            assert "functions" in result.structured_content
            assert result.structured_content["summary"]["average_complexity"] > 0
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_analyze_complexity_non_python(self, mcp_server):
        """Test analyzing non-Python file raises error."""
        tool = await mcp_server.get_tool("analyze_complexity")
        
        with pytest.raises(ToolError) as exc_info:
            await tool.fn(file_path="test.js")
        assert "Only Python files" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_monitoring_status(self, mcp_server, mock_repository_monitor):
        """Test getting monitoring status."""
        # Set up monitoring state
        mock_repository_monitor.is_running = True
        mock_repository_monitor.pending_changes = [
            {"path": "file1.py", "change_type": "modified"},
            {"path": "file2.py", "change_type": "added"}
        ]
        mock_repository_monitor.recent_changes = [
            {"path": "file3.py", "change_type": "deleted", "timestamp": datetime.now()}
        ]
        
        tool = await mcp_server.get_tool("monitoring_status")
        result = await tool.fn()
        
        assert result is not None
        status = result.structured_content
        assert status["is_running"] is True
        assert status["repository_path"] == "/test/repo"
        assert len(status["file_patterns"]) > 0
        assert status["pending_changes"] == 2
        assert len(status["recent_changes"]) == 2  # Should match pending_changes count (up to 5)
        assert "version_info" in status

    @pytest.mark.asyncio
    async def test_monitoring_status_not_running(self, mcp_server, mock_repository_monitor):
        """Test monitoring status when not running."""
        mock_repository_monitor.is_running = False
        mock_repository_monitor.monitoring_since = None
        
        tool = await mcp_server.get_tool("monitoring_status")
        result = await tool.fn()
        
        assert result.structured_content["is_running"] is False
        assert result.structured_content["monitoring_since"] is None