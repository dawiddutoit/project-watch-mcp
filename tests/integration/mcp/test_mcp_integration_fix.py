"""Test that verifies initialize_repository tool works with new background indexing."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, MagicMock, patch

import pytest

from src.project_watch_mcp.server import create_mcp_server


@pytest.mark.asyncio
async def test_initialize_repository_background_indexing():
    """Test that initialize_repository starts background indexing and returns immediately."""
    
    # Create mock repository monitor
    mock_monitor = AsyncMock()
    mock_monitor.is_running = False
    
    # Create mock files with mocked Path objects
    mock_files = []
    file_contents = {}
    for i in range(5):
        mock_file = MagicMock()
        # Use string path for simplicity
        file_path = f"/test/file_{i}.py"
        mock_file.path = MagicMock()
        mock_file.path.__str__ = Mock(return_value=file_path)
        # Store content for our mock
        file_contents[file_path] = f"# Test file {i}\nprint('hello')"
        # Create a mock that returns the right content
        mock_file.path.read_text = Mock(return_value=file_contents[file_path])
        mock_file.language = "python"
        mock_file.size = 100
        mock_file.last_modified = 123456789
        mock_files.append(mock_file)
    
    mock_monitor.scan_repository = AsyncMock(return_value=mock_files)
    mock_monitor.start = AsyncMock()
    
    # Create mock Neo4j RAG
    mock_rag = AsyncMock()
    # Track if index_file was called (proves background task ran)
    index_file_called = []
    
    async def track_index_file(code_file):
        index_file_called.append(code_file)
        # Add a small delay to simulate real indexing
        await asyncio.sleep(0.001)
    
    mock_rag.index_file = AsyncMock(side_effect=track_index_file)
    
    # Create MCP server
    server = create_mcp_server(
        repository_monitor=mock_monitor,
        neo4j_rag=mock_rag,
        project_name="test-project"
    )
    
    # Get the initialize_repository tool
    tool = await server.get_tool("initialize_repository")
    assert tool is not None
    
    # Execute the tool
    result = await tool.run({})
    
    # Verify result structure matches new background indexing format
    assert result is not None
    assert hasattr(result, "structured_content")
    
    # Check new structure
    assert result.structured_content["status"] == "indexing_started"
    assert result.structured_content["total"] == 5
    assert "background" in result.structured_content["message"].lower()
    
    # Verify monitoring was started
    mock_monitor.start.assert_called_once()
    
    # Give background task time to complete (it has 5 files with 0.001s delay each)
    await asyncio.sleep(0.2)
    
    # Verify that indexing actually happened in the background
    # We can't always catch the task itself, but we can verify it ran
    # by checking if index_file was called
    assert len(index_file_called) > 0, "Background indexing should have indexed at least some files"
    
    # Clean up any remaining background tasks
    all_tasks = asyncio.all_tasks()
    background_tasks = [t for t in all_tasks if t.get_name() == "background-indexing"]
    for task in background_tasks:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass