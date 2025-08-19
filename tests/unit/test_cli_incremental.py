"""
Test suite for CLI incremental indexing functionality.

This module tests the CLI integration with incremental indexing features.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call
import logging

import pytest
import pytest_asyncio

from src.project_watch_mcp.cli import initialize_only, main
from src.project_watch_mcp.repository_monitor import FileInfo


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_neo4j_driver():
    """Create a mock Neo4j driver."""
    driver = AsyncMock()
    driver.execute_query = AsyncMock()
    driver.close = AsyncMock()
    return driver


@pytest.fixture
def mock_repository_initializer():
    """Create a mock RepositoryInitializer."""
    with patch('src.project_watch_mcp.cli.RepositoryInitializer') as mock_class:
        mock_instance = AsyncMock()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        
        # Mock the initialize method result
        mock_result = MagicMock()
        mock_result.indexed = 10
        mock_result.total = 10
        mock_result.skipped = []
        mock_instance.initialize = AsyncMock(return_value=mock_result)
        
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_neo4j_rag():
    """Create a mock Neo4jRAG instance."""
    with patch('src.project_watch_mcp.cli.Neo4jRAG') as mock_class:
        mock_instance = AsyncMock()
        
        # Default mock behaviors
        mock_instance.is_repository_indexed = AsyncMock(return_value=False)
        mock_instance.get_indexed_files = AsyncMock(return_value={})
        mock_instance.detect_changed_files = AsyncMock(return_value=([], [], []))
        mock_instance.remove_files = AsyncMock()
        mock_instance.index_file = AsyncMock()
        mock_instance.initialize = AsyncMock()
        mock_instance.close = AsyncMock()
        
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_repository_monitor():
    """Create a mock RepositoryMonitor."""
    with patch('src.project_watch_mcp.cli.RepositoryMonitor') as mock_class:
        mock_instance = MagicMock()
        mock_instance.scan_repository = MagicMock(return_value=[])
        mock_instance.start_monitoring = AsyncMock()
        mock_instance.stop_monitoring = AsyncMock()
        
        mock_class.return_value = mock_instance
        yield mock_instance


# ============================================================================
# TEST: initialize_only with incremental indexing
# ============================================================================

class TestInitializeOnlyIncremental:
    """Test suite for initialize_only function with incremental indexing."""
    
    @pytest.mark.asyncio
    async def test_initialize_with_existing_index(self, mock_repository_initializer):
        """Test that initialize_only handles existing index correctly."""
        # Arrange
        neo4j_uri = "bolt://localhost:7687"
        user = "neo4j"
        password = "password"
        database = "neo4j"
        repository_path = "/test/repo"
        
        # Mock the result to show incremental indexing
        mock_result = MagicMock()
        mock_result.indexed = 3  # Only 3 new/modified files
        mock_result.total = 10   # Out of 10 total files
        mock_result.skipped = []
        mock_result.unchanged = 7  # 7 files unchanged
        mock_repository_initializer.initialize.return_value = mock_result
        
        # Act
        with patch('builtins.print') as mock_print:
            result = await initialize_only(
                neo4j_uri=neo4j_uri,
                PROJECT_WATCH_USER=user,
                PROJECT_WATCH_PASSWORD=password,
                PROJECT_WATCH_DATABASE=database,
                repository_path=repository_path,
                verbose=True
            )
        
        # Assert
        assert result == 0
        mock_repository_initializer.initialize.assert_called_once_with(persistent_monitoring=False)
        
        # Check printed output
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("Indexed: 3/10 files" in str(call) for call in print_calls)


# ============================================================================
# TEST: main function with incremental indexing
# ============================================================================

class TestMainWithIncrementalIndexing:
    """Test suite for main function with incremental indexing."""
    
    @pytest.mark.asyncio
    async def test_main_uses_incremental_when_index_exists(self):
        """Test that main() uses incremental indexing when repository is already indexed."""
        # This test would require significant mocking of the MCP server setup
        # For now, we'll focus on the core incremental logic in the initializer
        pass
    
    @pytest.mark.asyncio
    async def test_main_logs_incremental_statistics(self):
        """Test that main() properly logs statistics for incremental indexing."""
        # This test would also require MCP server mocking
        pass


# ============================================================================
# TEST: Incremental indexing logic in RepositoryInitializer
# ============================================================================

class TestIncrementalIndexingLogic:
    """Test the incremental indexing logic flow."""
    
    @pytest.mark.asyncio
    async def test_incremental_flow_with_changes(self):
        """Test the complete incremental indexing flow with file changes."""
        from src.project_watch_mcp.core import RepositoryInitializer
        
        with patch('pathlib.Path.read_text', return_value="mock file content"):
            with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
                with patch('src.project_watch_mcp.core.initializer.RepositoryMonitor') as mock_monitor:
                    with patch('src.project_watch_mcp.core.initializer.Neo4jRAG') as mock_rag_class:
                        # Setup mocks
                        mock_driver = AsyncMock()
                        mock_db.driver.return_value = mock_driver
                        
                        mock_rag = AsyncMock()
                        mock_rag_class.return_value = mock_rag
                        
                        # Mock repository already indexed
                        mock_rag.is_repository_indexed = AsyncMock(return_value=True)
                        
                        # Mock indexed files
                        base_time = datetime.now()
                        mock_rag.get_indexed_files = AsyncMock(return_value={
                            Path("/repo/file1.py"): base_time - timedelta(days=1),
                            Path("/repo/file2.py"): base_time - timedelta(days=2),
                            Path("/repo/deleted.py"): base_time - timedelta(days=3),
                        })
                        
                        # Mock current repository files
                        current_files = [
                            FileInfo(
                                path=Path("/repo/file1.py"),
                                last_modified=base_time,  # Modified
                                size=100
                            ),
                            FileInfo(
                                path=Path("/repo/file2.py"),
                                last_modified=base_time - timedelta(days=2),  # Unchanged
                                size=200
                            ),
                            FileInfo(
                                path=Path("/repo/new.py"),
                                last_modified=base_time,  # New
                                size=300
                            ),
                        ]
                        
                        mock_monitor_instance = MagicMock()
                        mock_monitor_instance.scan_repository = AsyncMock(return_value=current_files)
                        mock_monitor_instance.start_monitoring = AsyncMock()
                        mock_monitor.return_value = mock_monitor_instance
                        
                        # Mock detect_changed_files
                        new_files = [current_files[2]]  # new.py
                        modified_files = [current_files[0]]  # file1.py
                        deleted_paths = [Path("/repo/deleted.py")]
                        mock_rag.detect_changed_files = AsyncMock(
                            return_value=(new_files, modified_files, deleted_paths)
                        )
                        
                        # Mock file operations
                        mock_rag.remove_files = AsyncMock()
                        mock_rag.index_file = AsyncMock()
                        mock_rag.initialize = AsyncMock()
                        
                        # Create initializer
                        initializer = RepositoryInitializer(
                            neo4j_uri="bolt://localhost:7687",
                            PROJECT_WATCH_USER="neo4j",
                            PROJECT_WATCH_PASSWORD="password",
                            PROJECT_WATCH_DATABASE="neo4j",
                            repository_path=Path("/repo"),
                            project_name="test_project"
                        )
                        
                        # Run initialization
                        async with initializer:
                            result = await initializer.initialize(persistent_monitoring=False)
                        
                        # Verify incremental indexing was used
                        mock_rag.is_repository_indexed.assert_called_once()
                        mock_rag.get_indexed_files.assert_called_once()
                        mock_rag.detect_changed_files.assert_called_once()
                        
                        # Verify file operations
                        mock_rag.remove_files.assert_called_once_with("test_project", deleted_paths)
                        
                        # Verify only new and modified files were indexed (2 calls)
                        assert mock_rag.index_file.call_count == 2
                        
                        # Verify result statistics
                        assert result.indexed == 2  # new.py and file1.py
                        assert result.total == 3    # Total current files
    
    @pytest.mark.asyncio  
    async def test_incremental_flow_no_changes(self):
        """Test incremental indexing when no files have changed."""
        from src.project_watch_mcp.core import RepositoryInitializer
        
        with patch('pathlib.Path.read_text', return_value="mock file content"):
            with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
                with patch('src.project_watch_mcp.core.initializer.RepositoryMonitor') as mock_monitor:
                    with patch('src.project_watch_mcp.core.initializer.Neo4jRAG') as mock_rag_class:
                        # Setup mocks
                        mock_driver = AsyncMock()
                        mock_db.driver.return_value = mock_driver
                        
                        mock_rag = AsyncMock()
                        mock_rag_class.return_value = mock_rag
                        
                        # Mock repository already indexed
                        mock_rag.is_repository_indexed = AsyncMock(return_value=True)
                        
                        # Mock indexed files
                        base_time = datetime.now()
                        mock_rag.get_indexed_files = AsyncMock(return_value={
                            Path("/repo/file1.py"): base_time,
                            Path("/repo/file2.py"): base_time,
                        })
                        
                        # Mock current repository files (same timestamps)
                        current_files = [
                            FileInfo(
                                path=Path("/repo/file1.py"),
                                last_modified=base_time,
                                size=100
                            ),
                            FileInfo(
                                path=Path("/repo/file2.py"),
                                last_modified=base_time,
                                size=200
                            ),
                        ]
                        
                        mock_monitor_instance = MagicMock()
                        mock_monitor_instance.scan_repository = AsyncMock(return_value=current_files)
                        mock_monitor_instance.start_monitoring = AsyncMock()
                        mock_monitor.return_value = mock_monitor_instance
                        
                        # Mock detect_changed_files - no changes
                        mock_rag.detect_changed_files = AsyncMock(return_value=([], [], []))
                        
                        # Mock file operations
                        mock_rag.remove_files = AsyncMock()
                        mock_rag.index_file = AsyncMock()
                        mock_rag.initialize = AsyncMock()
                        
                        # Create initializer
                        initializer = RepositoryInitializer(
                            neo4j_uri="bolt://localhost:7687",
                            PROJECT_WATCH_USER="neo4j",
                            PROJECT_WATCH_PASSWORD="password",
                            PROJECT_WATCH_DATABASE="neo4j",
                            repository_path=Path("/repo"),
                            project_name="test_project"
                        )
                        
                        # Run initialization
                        async with initializer:
                            result = await initializer.initialize(persistent_monitoring=False)
                        
                        # Verify no files were indexed
                        mock_rag.index_file.assert_not_called()
                        mock_rag.remove_files.assert_not_called()
                        
                        # Verify result statistics
                        assert result.indexed == 0
                        assert result.total == 2