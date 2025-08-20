"""Test line counting functionality in file indexing."""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neo4j import AsyncDriver, RoutingControl

from project_watch_mcp.neo4j_rag import CodeFile, Neo4jRAG


@pytest.fixture
def mock_driver():
    """Create a mock Neo4j driver."""
    driver = MagicMock(spec=AsyncDriver)
    driver.execute_query = AsyncMock()
    return driver


@pytest.fixture
def neo4j_rag(mock_driver):
    """Create a Neo4jRAG instance with mocked driver."""
    return Neo4jRAG(
        neo4j_driver=mock_driver,
        project_name="test_project",
        embeddings=None
    )


@pytest.mark.asyncio
async def test_index_file_stores_line_count(neo4j_rag, mock_driver):
    """Test that indexing a file correctly stores the line count."""
    # Create a test file with known line count
    test_content = """line 1
line 2
line 3
line 4
line 5"""
    
    code_file = CodeFile(
        project_name="test_project",
        path=Path("/test/file.py"),
        content=test_content,
        language="python",
        size=len(test_content),
        last_modified=datetime.now()
    )
    
    # Index the file
    await neo4j_rag.index_file(code_file)
    
    # Verify that execute_query was called with the correct line count
    calls = mock_driver.execute_query.call_args_list
    
    # Find the file creation/update call
    file_create_call = None
    for call in calls:
        if "MERGE (f:CodeFile" in call[0][0]:
            file_create_call = call
            break
    
    assert file_create_call is not None, "File creation query not found"
    
    # Check that lines parameter is 5
    params = file_create_call[0][1]
    assert params["lines"] == 5, f"Expected 5 lines, got {params['lines']}"


@pytest.mark.asyncio
async def test_get_file_metadata_returns_line_count(neo4j_rag, mock_driver):
    """Test that get_file_metadata returns the line count."""
    # Mock the query result
    mock_record = {
        "path": "/test/file.py",
        "language": "python",
        "size": 1000,
        "lines": 42,  # The line count we expect
        "last_modified": "2024-01-01T00:00:00",
        "hash": "abc123",
        "project_name": "test_project",
        "chunk_count": 3
    }
    
    mock_result = MagicMock()
    mock_result.records = [mock_record]
    mock_driver.execute_query.return_value = mock_result
    
    # Get file metadata
    metadata = await neo4j_rag.get_file_metadata(Path("/test/file.py"))
    
    # Verify line count is returned
    assert metadata is not None
    assert metadata["lines"] == 42


@pytest.mark.asyncio
async def test_get_file_metadata_handles_missing_line_count(neo4j_rag, mock_driver):
    """Test that get_file_metadata handles files without line count gracefully."""
    # Mock the query result without lines field (old data)
    mock_record_data = {
        "path": "/test/file.py",
        "language": "python",
        "size": 1000,
        # "lines" key is missing entirely - simulating old data
        "last_modified": "2024-01-01T00:00:00",
        "hash": "abc123",
        "chunk_count": 3,
        "project_name": "test_project"
    }
    
    class MockRecord:
        def __init__(self, data):
            self.data = data
        
        def __getitem__(self, key):
            return self.data[key]
        
        def get(self, key, default=None):
            return self.data.get(key, default)
    
    mock_record_obj = MockRecord(mock_record_data)
    
    mock_result = MagicMock()
    mock_result.records = [mock_record_obj]
    mock_driver.execute_query.return_value = mock_result
    
    # Get file metadata
    metadata = await neo4j_rag.get_file_metadata(Path("/test/file.py"))
    
    # Verify line count defaults to 0
    assert metadata is not None
    assert metadata["lines"] == 0


@pytest.mark.asyncio
async def test_empty_file_has_zero_lines(neo4j_rag, mock_driver):
    """Test that an empty file has 0 lines."""
    code_file = CodeFile(
        project_name="test_project",
        path=Path("/test/empty.py"),
        content="",
        language="python",
        size=0,
        last_modified=datetime.now()
    )
    
    # Index the file
    await neo4j_rag.index_file(code_file)
    
    # Verify that execute_query was called with lines=0
    calls = mock_driver.execute_query.call_args_list
    
    # Find the file creation/update call
    file_create_call = None
    for call in calls:
        if "MERGE (f:CodeFile" in call[0][0]:
            file_create_call = call
            break
    
    assert file_create_call is not None
    params = file_create_call[0][1]
    assert params["lines"] == 0


@pytest.mark.asyncio
async def test_multiline_file_counts_correctly(neo4j_rag, mock_driver):
    """Test that files with various line endings are counted correctly."""
    # Test with Unix line endings
    unix_content = "line1\nline2\nline3"
    code_file = CodeFile(
        project_name="test_project",
        path=Path("/test/unix.py"),
        content=unix_content,
        language="python",
        size=len(unix_content),
        last_modified=datetime.now()
    )
    
    await neo4j_rag.index_file(code_file)
    
    # Get the last call
    calls = mock_driver.execute_query.call_args_list
    file_create_call = None
    for call in reversed(calls):
        if "MERGE (f:CodeFile" in call[0][0]:
            file_create_call = call
            break
    
    assert file_create_call is not None
    params = file_create_call[0][1]
    assert params["lines"] == 3
    
    # Test with Windows line endings
    windows_content = "line1\r\nline2\r\nline3"
    code_file = CodeFile(
        project_name="test_project",
        path=Path("/test/windows.py"),
        content=windows_content,
        language="python",
        size=len(windows_content),
        last_modified=datetime.now()
    )
    
    await neo4j_rag.index_file(code_file)
    
    # Get the last call again
    calls = mock_driver.execute_query.call_args_list
    file_create_call = None
    for call in reversed(calls):
        if "MERGE (f:CodeFile" in call[0][0]:
            file_create_call = call
            break
    
    assert file_create_call is not None
    params = file_create_call[0][1]
    # splitlines() handles both Unix and Windows line endings correctly
    assert params["lines"] == 3