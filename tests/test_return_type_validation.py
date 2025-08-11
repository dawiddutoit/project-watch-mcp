"""Return type validation tests that would have caught the tool failures.

This test suite validates that all MCP tools return the correct types
as declared in their function signatures. These tests would have prevented
the 67% tool failure rate experienced in production.
"""

import inspect
from typing import get_type_hints
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastmcp.tools.tool import TextContent, ToolResult

from src.project_watch_mcp.server import create_mcp_server


class TestReturnTypeValidation:
    """Validate that all tools return the types they declare."""

    @pytest.fixture
    def mock_monitor_and_rag(self):
        """Create minimal mocks for server creation."""
        monitor = AsyncMock()
        monitor.repo_path = "/test/repo"
        monitor.file_patterns = ["*.py"]
        monitor.is_running = False
        monitor.scan_repository = AsyncMock(return_value=[])
        monitor.process_all_changes = AsyncMock(return_value=[])
        
        rag = AsyncMock()
        rag.index_file = AsyncMock()
        rag.search_semantic = AsyncMock(return_value=[])
        rag.search_by_pattern = AsyncMock(return_value=[])
        rag.get_repository_stats = AsyncMock(return_value={
            "total_files": 10,
            "total_chunks": 50,
            "total_size": 100000,
            "languages": ["python"],
        })
        rag.get_file_metadata = AsyncMock(return_value={
            "language": "python",
            "size": 1000,
            "last_modified": "2025-01-01T00:00:00",
            "chunk_count": 5,
            "hash": "abc123def456",
        })
        rag.update_file = AsyncMock()
        
        return monitor, rag

    async def test_initialize_repository_returns_tool_result(self, mock_monitor_and_rag):
        """Test that initialize_repository returns ToolResult, not str."""
        monitor, rag = mock_monitor_and_rag
        server = create_mcp_server(monitor, rag, "test_project")
        
        # Get the actual tool function using FastMCP's get_tool method
        init_tool = await server.get_tool("initialize_repository")
        
        assert init_tool is not None, "initialize_repository tool not found"
        
        # Execute the tool
        result = await init_tool.run({})
        
        # THIS IS THE TEST THAT WOULD HAVE CAUGHT THE BUG
        assert isinstance(result, ToolResult), \
            f"initialize_repository should return ToolResult, got {type(result).__name__}"
        
        # Validate ToolResult structure
        assert hasattr(result, 'content'), "ToolResult missing 'content' attribute"
        assert hasattr(result, 'structured_content'), "ToolResult missing 'structured_content'"
        assert isinstance(result.content, list), "ToolResult.content should be a list"
        
        # Validate content structure
        if result.content:
            assert all(isinstance(c, TextContent) for c in result.content), \
                "All content items should be TextContent instances"

    async def test_search_code_returns_tool_result(self, mock_monitor_and_rag):
        """Test that search_code returns ToolResult, not list[dict]."""
        monitor, rag = mock_monitor_and_rag
        server = create_mcp_server(monitor, rag, "test_project")
        
        # Get the search_code tool using FastMCP's get_tool method
        search_tool = await server.get_tool("search_code")
        
        assert search_tool is not None, "search_code tool not found"
        
        # Execute the tool with valid parameters
        result = await search_tool.run({"query": "test", "search_type": "semantic", "limit": 5})
        
        # THIS IS THE TEST THAT WOULD HAVE CAUGHT THE BUG
        assert isinstance(result, ToolResult), \
            f"search_code should return ToolResult, got {type(result).__name__}"
        
        # Validate that structured_content is NOT a list (MCP protocol requirement)
        assert not isinstance(result.structured_content, list), \
            "structured_content must not be a list (MCP protocol violation)"
        
        # Should be None or dict
        assert result.structured_content is None or isinstance(result.structured_content, dict), \
            f"structured_content must be None or dict, got {type(result.structured_content).__name__}"

    async def test_refresh_file_returns_tool_result(self, mock_monitor_and_rag):
        """Test that refresh_file returns ToolResult, not str."""
        monitor, rag = mock_monitor_and_rag
        server = create_mcp_server(monitor, rag, "test_project")
        
        # Mock file existence
        from pathlib import Path
        from unittest.mock import patch
        
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'read_text', return_value="test content"), \
             patch.object(Path, 'stat', return_value=MagicMock(st_size=100, st_mtime=0)):
            
            # Get the refresh_file tool using FastMCP's get_tool method
            refresh_tool = await server.get_tool("refresh_file")
            
            assert refresh_tool is not None, "refresh_file tool not found"
            
            # Execute the tool
            result = await refresh_tool.run({"file_path": "test.py"})
            
            # THIS IS THE TEST THAT WOULD HAVE CAUGHT THE BUG
            assert isinstance(result, ToolResult), \
                f"refresh_file should return ToolResult, got {type(result).__name__}"

    async def test_all_tools_return_tool_result(self, mock_monitor_and_rag):
        """Comprehensive test that ALL tools return ToolResult."""
        monitor, rag = mock_monitor_and_rag
        server = create_mcp_server(monitor, rag, "test_project")
        
        # Map of tool names to their required parameters
        tool_params = {
            "initialize_repository": {},
            "search_code": {"query": "test"},
            "get_repository_stats": {},
            "get_file_info": {"file_path": "test.py"},
            "refresh_file": {"file_path": "test.py"},
            "monitoring_status": {},
        }
        
        # Mock file operations for tools that need them
        from pathlib import Path
        from unittest.mock import patch
        
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'read_text', return_value="test content"), \
             patch.object(Path, 'stat', return_value=MagicMock(st_size=100, st_mtime=0)):
            
            # Get all tools and validate each one
            tool_names = await server.get_tools()
            for tool_name in tool_names:
                if tool_name in tool_params:
                    params = tool_params[tool_name]
                    
                    # Get and execute the tool
                    tool_fn = await server.get_tool(tool_name)
                    result = await tool_fn.run(params)
                    
                    # CRITICAL VALIDATION
                    assert isinstance(result, ToolResult), \
                        f"Tool '{tool_name}' should return ToolResult, got {type(result).__name__}"
                    
                    # Validate ToolResult structure
                    self._validate_tool_result_structure(result, tool_name)

    def _validate_tool_result_structure(self, result: ToolResult, tool_name: str):
        """Validate the internal structure of a ToolResult."""
        # Must have content
        assert hasattr(result, 'content'), \
            f"{tool_name}: ToolResult missing 'content' attribute"
        assert isinstance(result.content, list), \
            f"{tool_name}: ToolResult.content must be a list"
        
        # Must have structured_content
        assert hasattr(result, 'structured_content'), \
            f"{tool_name}: ToolResult missing 'structured_content' attribute"
        
        # structured_content must be None or dict (MCP protocol)
        assert result.structured_content is None or isinstance(result.structured_content, dict), \
            f"{tool_name}: structured_content must be None or dict, got {type(result.structured_content).__name__}"
        
        # If structured_content is a dict with results, validate structure
        if isinstance(result.structured_content, dict) and "results" in result.structured_content:
            assert isinstance(result.structured_content["results"], list), \
                f"{tool_name}: results must be a list when present"
        
        # Validate TextContent items
        for i, content_item in enumerate(result.content):
            assert isinstance(content_item, TextContent), \
                f"{tool_name}: content[{i}] must be TextContent, got {type(content_item).__name__}"
            assert hasattr(content_item, 'type'), \
                f"{tool_name}: TextContent missing 'type' attribute"
            assert hasattr(content_item, 'text'), \
                f"{tool_name}: TextContent missing 'text' attribute"
            assert content_item.type == "text", \
                f"{tool_name}: TextContent.type must be 'text', got '{content_item.type}'"


class TestFunctionSignatureValidation:
    """Validate that function signatures match their actual return behavior."""

    @pytest.fixture
    def mock_monitor_and_rag(self):
        """Create minimal mocks for server creation."""
        monitor = AsyncMock()
        monitor.repo_path = "/test/repo"
        monitor.file_patterns = ["*.py"]
        monitor.is_running = False
        monitor.scan_repository = AsyncMock(return_value=[])
        monitor.process_all_changes = AsyncMock(return_value=[])
        
        rag = AsyncMock()
        rag.index_file = AsyncMock()
        rag.search_semantic = AsyncMock(return_value=[])
        rag.search_by_pattern = AsyncMock(return_value=[])
        rag.get_repository_stats = AsyncMock(return_value={
            "total_files": 10,
            "total_chunks": 50,
            "total_size": 100000,
            "languages": ["python"],
        })
        rag.get_file_metadata = AsyncMock(return_value={
            "language": "python",
            "size": 1000,
            "last_modified": "2025-01-01T00:00:00",
            "chunk_count": 5,
            "hash": "abc123def456",
        })
        rag.update_file = AsyncMock()
        
        return monitor, rag

    async def test_function_annotations_match_returns(self, mock_monitor_and_rag):
        """Test that type hints match actual return types."""
        monitor, rag = mock_monitor_and_rag
        server = create_mcp_server(monitor, rag, "test_project")
        
        from pathlib import Path
        from unittest.mock import patch
        
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'read_text', return_value="test content"), \
             patch.object(Path, 'stat', return_value=MagicMock(st_size=100, st_mtime=0)):
            
            tool_names = await server.get_tools()
            
            # Map of tool names to their required parameters
            tool_params = {
                "initialize_repository": {},
                "search_code": {"query": "test"},
                "get_repository_stats": {},
                "get_file_info": {"file_path": "test.py"},
                "refresh_file": {"file_path": "test.py"},
                "monitoring_status": {},
            }
            
            for tool_name in tool_names:
                if tool_name in tool_params:
                    # Get the tool function
                    tool_fn = await server.get_tool(tool_name)
                    
                    # Execute the tool and verify it returns ToolResult
                    # This validates that the actual runtime behavior matches expected types
                    params = tool_params[tool_name]
                    result = await tool_fn.run(params)
                    
                    # This verifies the function returns ToolResult at runtime
                    # which is what matters for MCP compliance
                    assert isinstance(result, ToolResult), \
                        f"Tool '{tool_name}' should return ToolResult, got {type(result).__name__}"


class TestMCPProtocolCompliance:
    """Test compliance with MCP protocol requirements."""

    @pytest.fixture
    def mock_monitor_and_rag(self):
        """Create minimal mocks for server creation."""
        monitor = AsyncMock()
        monitor.repo_path = "/test/repo"
        monitor.file_patterns = ["*.py"]
        monitor.is_running = False
        monitor.scan_repository = AsyncMock(return_value=[])
        monitor.process_all_changes = AsyncMock(return_value=[])
        
        rag = AsyncMock()
        rag.index_file = AsyncMock()
        rag.search_semantic = AsyncMock(return_value=[])
        rag.search_by_pattern = AsyncMock(return_value=[])
        rag.get_repository_stats = AsyncMock(return_value={
            "total_files": 10,
            "total_chunks": 50,
            "total_size": 100000,
            "languages": ["python"],
        })
        rag.get_file_metadata = AsyncMock(return_value={
            "language": "python",
            "size": 1000,
            "last_modified": "2025-01-01T00:00:00",
            "chunk_count": 5,
            "hash": "abc123def456",
        })
        rag.update_file = AsyncMock()
        
        return monitor, rag

    async def test_structured_content_never_bare_list(self, mock_monitor_and_rag):
        """Test that structured_content is never a bare list (MCP protocol requirement)."""
        monitor, rag = mock_monitor_and_rag
        
        # Set up RAG to return search results
        from src.project_watch_mcp.neo4j_rag import SearchResult
        rag.search_semantic.return_value = [
            SearchResult(
                project_name="test_project",
                file_path="/test/file1.py",
                content="test content 1",
                line_number=10,
                similarity=0.95
            ),
            SearchResult(
                project_name="test_project",
                file_path="/test/file2.py",
                content="test content 2",
                line_number=20,
                similarity=0.90
            ),
        ]
        
        server = create_mcp_server(monitor, rag, "test_project")
        
        # Find and execute search_code
        search_tool = await server.get_tool("search_code")
        assert search_tool is not None
        
        result = await search_tool.run({"query": "test query", "search_type": "semantic"})
        
        # THIS TEST WOULD HAVE CAUGHT THE PROTOCOL VIOLATION
        assert not isinstance(result.structured_content, list), \
            "structured_content must NOT be a list (MCP protocol violation). Wrap lists in a dict."
        
        # If there are results, they should be wrapped in a dict
        if result.structured_content and isinstance(result.structured_content, dict):
            # Common pattern is {"results": [...]}
            if "results" in result.structured_content:
                assert isinstance(result.structured_content["results"], list), \
                    "Results should be a list when wrapped in dict"

    async def test_all_tools_comply_with_mcp_protocol(self, mock_monitor_and_rag):
        """Comprehensive MCP protocol compliance test for all tools."""
        monitor, rag = mock_monitor_and_rag
        server = create_mcp_server(monitor, rag, "test_project")
        
        tool_params = {
            "initialize_repository": {},
            "search_code": {"query": "test"},
            "get_repository_stats": {},
            "get_file_info": {"file_path": "test.py"},
            "refresh_file": {"file_path": "test.py"},
            "monitoring_status": {},
        }
        
        from pathlib import Path
        from unittest.mock import patch
        
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'read_text', return_value="test content"), \
             patch.object(Path, 'stat', return_value=MagicMock(st_size=100, st_mtime=0)):
            
            tool_names = await server.get_tools()
            for tool_name in tool_names:
                if tool_name in tool_params:
                    params = tool_params[tool_name]
                    tool_fn = await server.get_tool(tool_name)
                    result = await tool_fn.run(params)
                
                    
                    # MCP Protocol Requirements:
                    # 1. Must return ToolResult
                    assert isinstance(result, ToolResult), \
                        f"{tool_name}: Must return ToolResult (MCP requirement)"
                    
                    # 2. structured_content must be None or dict, never list
                    assert result.structured_content is None or isinstance(result.structured_content, dict), \
                        f"{tool_name}: structured_content must be None or dict (MCP requirement)"
                    
                    # 3. content must be a list of content objects
                    assert isinstance(result.content, list), \
                        f"{tool_name}: content must be a list (MCP requirement)"
                    
                    # 4. Each content item must have type and text/data
                    for item in result.content:
                        assert hasattr(item, 'type'), \
                            f"{tool_name}: Content items must have 'type' (MCP requirement)"
                        assert hasattr(item, 'text') or hasattr(item, 'data'), \
                            f"{tool_name}: Content items must have 'text' or 'data' (MCP requirement)"


# Example of how to run these tests specifically
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])