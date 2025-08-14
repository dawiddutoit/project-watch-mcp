"""Unit tests for MonitoringManager module."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from project_watch_mcp.core.monitoring_manager import MonitoringManager


class TestMonitoringManager:
    """Test MonitoringManager class."""

    @pytest.fixture
    def mock_neo4j_rag(self):
        """Create mock Neo4jRAG instance."""
        mock = AsyncMock()
        mock.index_file = AsyncMock()
        mock.update_file = AsyncMock()
        mock.delete_file = AsyncMock()
        return mock

    @pytest.fixture
    def monitoring_manager(self, mock_neo4j_rag):
        """Create MonitoringManager instance with mocked dependencies."""
        manager = MonitoringManager(
            repository_path=Path("/test/repo"),
            neo4j_rag=mock_neo4j_rag,
            file_patterns=["*.py", "*.js"],
            exclude_patterns=["node_modules", "__pycache__"],
        )
        return manager

    def test_initialization(self, monitoring_manager):
        """Test MonitoringManager initialization."""
        assert monitoring_manager.repository_path == Path("/test/repo")
        assert monitoring_manager.file_patterns == ["*.py", "*.js"]
        assert monitoring_manager.exclude_patterns == ["node_modules", "__pycache__"]
        assert monitoring_manager._monitored_files == {}
        assert monitoring_manager._stop_event is not None

    def test_should_monitor_file(self, monitoring_manager):
        """Test file monitoring logic."""
        # Test matching patterns
        assert monitoring_manager._should_monitor(Path("test.py"))
        assert monitoring_manager._should_monitor(Path("app.js"))
        assert not monitoring_manager._should_monitor(Path("test.txt"))
        
        # Test exclude patterns
        assert not monitoring_manager._should_monitor(Path("node_modules/test.py"))
        assert not monitoring_manager._should_monitor(Path("__pycache__/test.py"))

    @pytest.mark.asyncio
    async def test_handle_file_created(self, monitoring_manager, mock_neo4j_rag):
        """Test handling file creation events."""
        file_path = Path("/test/repo/new_file.py")
        
        # Simulate file creation
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.read_text", return_value="print('hello')"):
                await monitoring_manager._handle_file_created(file_path)
        
        # Verify file was indexed
        mock_neo4j_rag.index_file.assert_called_once()
        assert str(file_path) in monitoring_manager._monitored_files

    @pytest.mark.asyncio
    async def test_handle_file_modified(self, monitoring_manager, mock_neo4j_rag):
        """Test handling file modification events."""
        file_path = Path("/test/repo/existing.py")
        
        # Add file to monitored files
        monitoring_manager._monitored_files[str(file_path)] = "old_hash"
        
        # Simulate file modification
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.read_text", return_value="print('modified')"):
                await monitoring_manager._handle_file_modified(file_path)
        
        # Verify file was updated
        mock_neo4j_rag.update_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_file_deleted(self, monitoring_manager, mock_neo4j_rag):
        """Test handling file deletion events."""
        file_path = Path("/test/repo/deleted.py")
        
        # Add file to monitored files
        monitoring_manager._monitored_files[str(file_path)] = "file_hash"
        
        # Simulate file deletion
        await monitoring_manager._handle_file_deleted(file_path)
        
        # Verify file was removed from index
        mock_neo4j_rag.delete_file.assert_called_once_with(str(file_path))
        assert str(file_path) not in monitoring_manager._monitored_files

    @pytest.mark.asyncio
    async def test_scan_existing_files(self, monitoring_manager):
        """Test scanning existing files in repository."""
        test_files = [
            Path("/test/repo/file1.py"),
            Path("/test/repo/file2.js"),
            Path("/test/repo/ignored.txt"),
        ]
        
        with patch.object(Path, "rglob") as mock_rglob:
            # Mock file discovery
            mock_rglob.side_effect = [
                [test_files[0]],  # *.py files
                [test_files[1]],  # *.js files
            ]
            
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.read_text", return_value="content"):
                    files = await monitoring_manager._scan_existing_files()
        
        # Should only return matching files
        assert len(files) == 2
        assert test_files[0] in files
        assert test_files[1] in files

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, monitoring_manager):
        """Test starting and stopping monitoring."""
        # Mock the monitoring loop
        with patch.object(monitoring_manager, "_monitor_loop", new_callable=AsyncMock):
            # Start monitoring
            task = asyncio.create_task(monitoring_manager.start())
            
            # Give it time to start
            await asyncio.sleep(0.1)
            
            # Stop monitoring
            await monitoring_manager.stop()
            
            # Ensure task completes
            await task
            
            assert monitoring_manager._stop_event.is_set()

    def test_calculate_file_hash(self, monitoring_manager):
        """Test file hash calculation."""
        content = "test content"
        hash1 = monitoring_manager._calculate_hash(content)
        hash2 = monitoring_manager._calculate_hash(content)
        hash3 = monitoring_manager._calculate_hash("different content")
        
        # Same content should produce same hash
        assert hash1 == hash2
        # Different content should produce different hash
        assert hash1 != hash3
        # Hash should be a string
        assert isinstance(hash1, str)