"""Test that initialize_repository tool runs indexing in the background."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock

import pytest
from fastmcp.tools.tool import ToolResult

from project_watch_mcp.server import create_mcp_server


@pytest.mark.asyncio
async def test_initialize_repository_runs_in_background():
    """Test that initialize_repository tool starts indexing in background and returns immediately."""
    
    # Create mock repository monitor
    mock_monitor = AsyncMock()
    mock_monitor.is_running = False
    
    # Create mock file info objects
    mock_files = []
    for i in range(100):  # Many files to simulate long indexing
        mock_file = Mock()
        mock_file.path = Path(f"/test/file_{i}.py")
        mock_file.language = "python"
        mock_file.size = 1000
        mock_file.last_modified = 123456789
        mock_files.append(mock_file)
    
    mock_monitor.scan_repository = AsyncMock(return_value=mock_files)
    mock_monitor.start = AsyncMock()
    
    # Create mock Neo4j RAG
    mock_rag = AsyncMock()
    
    # Make index_file slow to simulate real indexing
    async def slow_index(*args, **kwargs):
        await asyncio.sleep(0.01)  # Simulate slow indexing
    
    mock_rag.index_file = AsyncMock(side_effect=slow_index)
    
    # Create MCP server
    server = create_mcp_server(
        repository_monitor=mock_monitor,
        neo4j_rag=mock_rag,
        project_name="test-project"
    )
    
    # Get the initialize_repository tool
    tool = await server.get_tool("initialize_repository")
    
    assert tool is not None, "initialize_repository tool not found"
    
    # Measure execution time
    start_time = asyncio.get_event_loop().time()
    
    # Execute the tool
    result = await tool.run({})
    
    end_time = asyncio.get_event_loop().time()
    execution_time = end_time - start_time
    
    # Tool should return quickly (< 1 second) even with 100 files
    assert execution_time < 1.0, f"Tool took {execution_time:.2f}s, should be non-blocking"
    
    # Verify result structure
    assert isinstance(result, ToolResult)
    assert result.structured_content is not None
    
    # Should indicate that indexing is in progress
    assert "background" in result.structured_content.get("message", "").lower() or \
           "started" in result.structured_content.get("message", "").lower() or \
           result.structured_content.get("status") == "indexing_started", \
           "Tool should indicate background indexing has started"


@pytest.mark.asyncio
async def test_initialize_repository_status_check():
    """Test that we can check the status of background indexing."""
    
    # Create mock repository monitor
    mock_monitor = AsyncMock()
    mock_monitor.is_running = False
    mock_monitor.scan_repository = AsyncMock(return_value=[])
    mock_monitor.start = AsyncMock()
    
    # Create mock Neo4j RAG
    mock_rag = AsyncMock()
    
    # Create MCP server
    server = create_mcp_server(
        repository_monitor=mock_monitor,
        neo4j_rag=mock_rag,
        project_name="test-project"
    )
    
    # Check if indexing_status tool exists
    tools = await server.get_tools()
    assert "indexing_status" in tools, "indexing_status tool should exist to check background indexing"