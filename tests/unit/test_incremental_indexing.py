"""
Test suite for incremental indexing functionality.

This module tests the incremental indexing features added to Neo4jRAG
to optimize server startup by only indexing new/changed files.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Tuple

import pytest
import pytest_asyncio
from neo4j import AsyncSession

from src.project_watch_mcp.neo4j_rag import Neo4jRAG, CodeFile
from src.project_watch_mcp.repository_monitor import FileInfo


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_neo4j_driver():
    """Create a mock Neo4j driver for testing."""
    driver = AsyncMock()
    driver.execute_query = AsyncMock()
    return driver


@pytest.fixture
def neo4j_rag(mock_neo4j_driver):
    """Create a Neo4jRAG instance with mocked driver."""
    return Neo4jRAG(
        neo4j_driver=mock_neo4j_driver,
        project_name="test_project",
        embeddings=None,  # Disable embeddings for these tests
        chunk_size=100,
        chunk_overlap=20
    )


# ============================================================================
# TEST: is_repository_indexed (Task 003)
# ============================================================================

class TestIsRepositoryIndexed:
    """Test suite for is_repository_indexed method."""
    
    @pytest.mark.asyncio
    async def test_repository_indexed_returns_true(self, neo4j_rag, mock_neo4j_driver):
        """Test that is_repository_indexed returns True when files exist."""
        # Arrange
        mock_result = MagicMock()
        mock_result.records = [{"file_count": 10}]
        mock_neo4j_driver.execute_query.return_value = mock_result
        
        # Act
        result = await neo4j_rag.is_repository_indexed("test_project")
        
        # Assert
        assert result is True
        mock_neo4j_driver.execute_query.assert_called_once()
        query_args = mock_neo4j_driver.execute_query.call_args
        assert "test_project" in str(query_args)
        
    @pytest.mark.asyncio
    async def test_repository_not_indexed_returns_false(self, neo4j_rag, mock_neo4j_driver):
        """Test that is_repository_indexed returns False when no files exist."""
        # Arrange
        mock_result = MagicMock()
        mock_result.records = [{"file_count": 0}]
        mock_neo4j_driver.execute_query.return_value = mock_result
        
        # Act
        result = await neo4j_rag.is_repository_indexed("test_project")
        
        # Assert
        assert result is False
        
    @pytest.mark.asyncio
    async def test_repository_indexed_with_different_project_names(self, neo4j_rag, mock_neo4j_driver):
        """Test is_repository_indexed with different project names."""
        # Arrange
        mock_result_project1 = MagicMock()
        mock_result_project1.records = [{"file_count": 5}]
        
        mock_result_project2 = MagicMock()
        mock_result_project2.records = [{"file_count": 0}]
        
        mock_neo4j_driver.execute_query.side_effect = [
            mock_result_project1,
            mock_result_project2
        ]
        
        # Act
        result1 = await neo4j_rag.is_repository_indexed("project1")
        result2 = await neo4j_rag.is_repository_indexed("project2")
        
        # Assert
        assert result1 is True
        assert result2 is False
        assert mock_neo4j_driver.execute_query.call_count == 2


# ============================================================================
# TEST: get_indexed_files (Task 004)
# ============================================================================

class TestGetIndexedFiles:
    """Test suite for get_indexed_files method."""
    
    @pytest.mark.asyncio
    async def test_get_indexed_files_returns_file_map(self, neo4j_rag, mock_neo4j_driver):
        """Test that get_indexed_files returns a map of paths to timestamps."""
        # Arrange
        test_time = datetime.now()
        mock_result = MagicMock()
        mock_result.records = [
            {"path": "/project/file1.py", "last_modified": test_time.isoformat()},
            {"path": "/project/file2.py", "last_modified": (test_time - timedelta(days=1)).isoformat()},
            {"path": "/project/file3.py", "last_modified": (test_time - timedelta(hours=2)).isoformat()},
        ]
        mock_neo4j_driver.execute_query.return_value = mock_result
        
        # Act
        result = await neo4j_rag.get_indexed_files("test_project")
        
        # Assert
        assert isinstance(result, dict)
        assert len(result) == 3
        assert Path("/project/file1.py") in result
        assert Path("/project/file2.py") in result
        assert Path("/project/file3.py") in result
        assert isinstance(result[Path("/project/file1.py")], datetime)
        
    @pytest.mark.asyncio
    async def test_get_indexed_files_empty_repository(self, neo4j_rag, mock_neo4j_driver):
        """Test that get_indexed_files returns empty dict for empty repository."""
        # Arrange
        mock_result = MagicMock()
        mock_result.records = []
        mock_neo4j_driver.execute_query.return_value = mock_result
        
        # Act
        result = await neo4j_rag.get_indexed_files("test_project")
        
        # Assert
        assert result == {}
        
    @pytest.mark.asyncio
    async def test_get_indexed_files_handles_missing_timestamp(self, neo4j_rag, mock_neo4j_driver):
        """Test that get_indexed_files handles files with missing timestamps."""
        # Arrange
        test_time = datetime.now()
        mock_result = MagicMock()
        mock_result.records = [
            {"path": "/project/file1.py", "last_modified": test_time.isoformat()},
            {"path": "/project/file2.py", "last_modified": None},  # Missing timestamp
            {"path": "/project/file3.py", "last_modified": ""},  # Empty timestamp
        ]
        mock_neo4j_driver.execute_query.return_value = mock_result
        
        # Act
        result = await neo4j_rag.get_indexed_files("test_project")
        
        # Assert
        assert len(result) == 3
        assert Path("/project/file1.py") in result
        assert Path("/project/file2.py") in result
        assert Path("/project/file3.py") in result
        # Files with missing timestamps should get a default old timestamp
        assert result[Path("/project/file2.py")] < test_time
        assert result[Path("/project/file3.py")] < test_time


# ============================================================================
# TEST: detect_changed_files (Task 005)
# ============================================================================

class TestDetectChangedFiles:
    """Test suite for detect_changed_files method."""
    
    @pytest.mark.asyncio
    async def test_detect_new_files(self, neo4j_rag):
        """Test detection of new files not in the index."""
        # Arrange
        base_time = datetime.now()
        current_files = [
            FileInfo(
                path=Path("/project/new_file.py"),
                last_modified=base_time,
                size=100
            ),
            FileInfo(
                path=Path("/project/existing_file.py"),
                last_modified=base_time - timedelta(days=1),
                size=200
            ),
        ]
        indexed_files = {
            Path("/project/existing_file.py"): base_time - timedelta(days=1)
        }
        
        # Act
        new_files, modified_files, deleted_paths = await neo4j_rag.detect_changed_files(
            current_files, indexed_files
        )
        
        # Assert
        assert len(new_files) == 1
        assert new_files[0].path == Path("/project/new_file.py")
        assert len(modified_files) == 0
        assert len(deleted_paths) == 0
        
    @pytest.mark.asyncio
    async def test_detect_modified_files(self, neo4j_rag):
        """Test detection of modified files with newer timestamps."""
        # Arrange
        base_time = datetime.now()
        current_files = [
            FileInfo(
                path=Path("/project/file1.py"),
                last_modified=base_time,  # Newer than indexed
                size=100
            ),
            FileInfo(
                path=Path("/project/file2.py"),
                last_modified=base_time - timedelta(days=2),  # Same as indexed
                size=200
            ),
        ]
        indexed_files = {
            Path("/project/file1.py"): base_time - timedelta(days=1),  # Older
            Path("/project/file2.py"): base_time - timedelta(days=2),  # Same
        }
        
        # Act
        new_files, modified_files, deleted_paths = await neo4j_rag.detect_changed_files(
            current_files, indexed_files
        )
        
        # Assert
        assert len(new_files) == 0
        assert len(modified_files) == 1
        assert modified_files[0].path == Path("/project/file1.py")
        assert len(deleted_paths) == 0
        
    @pytest.mark.asyncio
    async def test_detect_deleted_files(self, neo4j_rag):
        """Test detection of deleted files."""
        # Arrange
        base_time = datetime.now()
        current_files = [
            FileInfo(
                path=Path("/project/file1.py"),
                last_modified=base_time,
                size=100
            ),
        ]
        indexed_files = {
            Path("/project/file1.py"): base_time,
            Path("/project/file2.py"): base_time - timedelta(days=1),  # Deleted
            Path("/project/file3.py"): base_time - timedelta(days=2),  # Deleted
        }
        
        # Act
        new_files, modified_files, deleted_paths = await neo4j_rag.detect_changed_files(
            current_files, indexed_files
        )
        
        # Assert
        assert len(new_files) == 0
        assert len(modified_files) == 0
        assert len(deleted_paths) == 2
        assert Path("/project/file2.py") in deleted_paths
        assert Path("/project/file3.py") in deleted_paths
        
    @pytest.mark.asyncio
    async def test_detect_unchanged_files_ignored(self, neo4j_rag):
        """Test that unchanged files are properly ignored."""
        # Arrange
        base_time = datetime.now()
        current_files = [
            FileInfo(
                path=Path("/project/file1.py"),
                last_modified=base_time,
                size=100
            ),
            FileInfo(
                path=Path("/project/file2.py"),
                last_modified=base_time,
                size=200
            ),
        ]
        indexed_files = {
            Path("/project/file1.py"): base_time,  # Same timestamp
            Path("/project/file2.py"): base_time,  # Same timestamp
        }
        
        # Act
        new_files, modified_files, deleted_paths = await neo4j_rag.detect_changed_files(
            current_files, indexed_files
        )
        
        # Assert
        assert len(new_files) == 0
        assert len(modified_files) == 0
        assert len(deleted_paths) == 0
        
    @pytest.mark.asyncio
    async def test_detect_all_change_types(self, neo4j_rag):
        """Test detection of all change types in one operation."""
        # Arrange
        base_time = datetime.now()
        current_files = [
            FileInfo(
                path=Path("/project/new.py"),
                last_modified=base_time,
                size=100
            ),
            FileInfo(
                path=Path("/project/modified.py"),
                last_modified=base_time,
                size=200
            ),
            FileInfo(
                path=Path("/project/unchanged.py"),
                last_modified=base_time - timedelta(days=1),
                size=300
            ),
        ]
        indexed_files = {
            Path("/project/modified.py"): base_time - timedelta(hours=1),  # Older
            Path("/project/unchanged.py"): base_time - timedelta(days=1),  # Same
            Path("/project/deleted.py"): base_time - timedelta(days=2),  # Missing from current
        }
        
        # Act
        new_files, modified_files, deleted_paths = await neo4j_rag.detect_changed_files(
            current_files, indexed_files
        )
        
        # Assert
        assert len(new_files) == 1
        assert new_files[0].path == Path("/project/new.py")
        assert len(modified_files) == 1
        assert modified_files[0].path == Path("/project/modified.py")
        assert len(deleted_paths) == 1
        assert Path("/project/deleted.py") in deleted_paths


# ============================================================================
# TEST: remove_files (Task 007)
# ============================================================================

class TestRemoveFiles:
    """Test suite for remove_files method."""
    
    @pytest.mark.asyncio
    async def test_remove_single_file(self, neo4j_rag, mock_neo4j_driver):
        """Test removal of a single file from index."""
        # Arrange
        file_paths = [Path("/project/file1.py")]
        mock_neo4j_driver.execute_query.return_value = MagicMock()
        
        # Act
        await neo4j_rag.remove_files("test_project", file_paths)
        
        # Assert
        mock_neo4j_driver.execute_query.assert_called_once()
        query_args = mock_neo4j_driver.execute_query.call_args
        assert "test_project" in str(query_args)
        assert "/project/file1.py" in str(query_args)
        
    @pytest.mark.asyncio
    async def test_remove_multiple_files(self, neo4j_rag, mock_neo4j_driver):
        """Test removal of multiple files from index."""
        # Arrange
        file_paths = [
            Path("/project/file1.py"),
            Path("/project/file2.py"),
            Path("/project/dir/file3.py"),
        ]
        mock_neo4j_driver.execute_query.return_value = MagicMock()
        
        # Act
        await neo4j_rag.remove_files("test_project", file_paths)
        
        # Assert
        # Should be called once per file or once with all files depending on implementation
        assert mock_neo4j_driver.execute_query.called
        
    @pytest.mark.asyncio
    async def test_remove_nonexistent_files_gracefully(self, neo4j_rag, mock_neo4j_driver):
        """Test that remove_files handles non-existent files gracefully."""
        # Arrange
        file_paths = [Path("/project/nonexistent.py")]
        mock_result = MagicMock()
        mock_result.records = []  # No records affected
        mock_neo4j_driver.execute_query.return_value = mock_result
        
        # Act
        # Should not raise an exception
        await neo4j_rag.remove_files("test_project", file_paths)
        
        # Assert
        mock_neo4j_driver.execute_query.assert_called()
        
    @pytest.mark.asyncio
    async def test_remove_empty_file_list(self, neo4j_rag, mock_neo4j_driver):
        """Test that remove_files handles empty file list."""
        # Arrange
        file_paths = []
        
        # Act
        await neo4j_rag.remove_files("test_project", file_paths)
        
        # Assert
        # Should not make any database calls for empty list
        mock_neo4j_driver.execute_query.assert_not_called()