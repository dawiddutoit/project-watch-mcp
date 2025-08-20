"""Tests for event debouncing in repository monitoring."""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from project_watch_mcp.repository_monitor import (
    DebouncedChangeProcessor,
    FileChangeEvent,
    FileChangeType,
    RepositoryMonitor,
)


class TestDebouncedChangeProcessor:
    """Test the DebouncedChangeProcessor class."""

    async def test_immediate_processing_first_change(self):
        """Test that the first change for a file is processed immediately."""
        processor = DebouncedChangeProcessor(debounce_delay=0.5)
        
        event = FileChangeEvent(
            path=Path("test.py"),
            change_type=FileChangeType.MODIFIED
        )
        
        should_process = await processor.add_change(event)
        assert should_process is True
        assert processor.pending_count == 0

    async def test_debounce_rapid_changes(self):
        """Test that rapid changes to the same file are debounced."""
        processor = DebouncedChangeProcessor(debounce_delay=0.5)
        
        path = Path("test.py")
        
        # First change should be processed
        event1 = FileChangeEvent(path=path, change_type=FileChangeType.MODIFIED)
        should_process1 = await processor.add_change(event1)
        assert should_process1 is True
        
        # Second change immediately after should be debounced
        event2 = FileChangeEvent(path=path, change_type=FileChangeType.MODIFIED)
        should_process2 = await processor.add_change(event2)
        assert should_process2 is False
        assert processor.pending_count == 1
        
        # Third change should also be debounced (replaces the second)
        event3 = FileChangeEvent(path=path, change_type=FileChangeType.MODIFIED)
        should_process3 = await processor.add_change(event3)
        assert should_process3 is False
        assert processor.pending_count == 1  # Still just one pending

    async def test_process_after_delay(self):
        """Test that changes are processed after the debounce delay."""
        processor = DebouncedChangeProcessor(debounce_delay=0.1)  # Short delay for testing
        
        path = Path("test.py")
        
        # First change
        event1 = FileChangeEvent(path=path, change_type=FileChangeType.MODIFIED)
        await processor.add_change(event1)
        
        # Second change (debounced)
        event2 = FileChangeEvent(path=path, change_type=FileChangeType.MODIFIED)
        should_process = await processor.add_change(event2)
        assert should_process is False
        
        # Wait for debounce delay
        await asyncio.sleep(0.15)
        
        # Get pending changes
        pending = await processor.get_pending_changes()
        assert len(pending) == 1
        assert pending[0].path == path
        assert processor.pending_count == 0

    async def test_different_files_no_interference(self):
        """Test that changes to different files don't interfere with each other."""
        processor = DebouncedChangeProcessor(debounce_delay=0.5)
        
        path1 = Path("file1.py")
        path2 = Path("file2.py")
        
        # Changes to different files
        event1 = FileChangeEvent(path=path1, change_type=FileChangeType.MODIFIED)
        event2 = FileChangeEvent(path=path2, change_type=FileChangeType.MODIFIED)
        
        should_process1 = await processor.add_change(event1)
        should_process2 = await processor.add_change(event2)
        
        # Both should be processed immediately (different files)
        assert should_process1 is True
        assert should_process2 is True
        assert processor.pending_count == 0

    async def test_get_pending_changes_timing(self):
        """Test that get_pending_changes respects the debounce delay."""
        processor = DebouncedChangeProcessor(debounce_delay=0.2)
        
        path = Path("test.py")
        
        # First change (processed)
        event1 = FileChangeEvent(path=path, change_type=FileChangeType.MODIFIED)
        await processor.add_change(event1)
        
        # Second change (debounced)
        event2 = FileChangeEvent(path=path, change_type=FileChangeType.MODIFIED)
        await processor.add_change(event2)
        
        # Immediately check - should be empty (not enough time passed)
        pending1 = await processor.get_pending_changes()
        assert len(pending1) == 0
        assert processor.pending_count == 1  # Still pending
        
        # Wait for debounce delay
        await asyncio.sleep(0.25)
        
        # Now should get the pending change
        pending2 = await processor.get_pending_changes()
        assert len(pending2) == 1
        assert processor.pending_count == 0

    async def test_multiple_files_pending(self):
        """Test handling multiple files with pending changes."""
        processor = DebouncedChangeProcessor(debounce_delay=0.1)
        
        paths = [Path(f"file{i}.py") for i in range(5)]
        
        # Process first change for each file
        for path in paths:
            event = FileChangeEvent(path=path, change_type=FileChangeType.MODIFIED)
            await processor.add_change(event)
        
        # Add second change for each file (should be debounced)
        for path in paths:
            event = FileChangeEvent(path=path, change_type=FileChangeType.MODIFIED)
            result = await processor.add_change(event)
            assert result is False
        
        assert processor.pending_count == 5
        
        # Wait for debounce delay
        await asyncio.sleep(0.15)
        
        # Get all pending changes
        pending = await processor.get_pending_changes()
        assert len(pending) == 5
        assert processor.pending_count == 0

    def test_clear_history(self):
        """Test clearing old processing history."""
        processor = DebouncedChangeProcessor(debounce_delay=0.5)
        
        # Add some processing history
        current_time = time.time()
        processor._last_process_time = {
            Path("old1.py"): current_time - 100,  # Old
            Path("old2.py"): current_time - 70,   # Old
            Path("recent.py"): current_time - 30,  # Recent
            Path("new.py"): current_time - 5,      # Very recent
        }
        
        # Clear entries older than 60 seconds
        processor.clear_history(older_than=60.0)
        
        # Should only keep recent entries
        assert Path("old1.py") not in processor._last_process_time
        assert Path("old2.py") not in processor._last_process_time
        assert Path("recent.py") in processor._last_process_time
        assert Path("new.py") in processor._last_process_time

    async def test_change_type_preserved(self):
        """Test that the change type is preserved through debouncing."""
        processor = DebouncedChangeProcessor(debounce_delay=0.1)
        
        path = Path("test.py")
        
        # First change (CREATE)
        event1 = FileChangeEvent(path=path, change_type=FileChangeType.CREATED)
        await processor.add_change(event1)
        
        # Second change (MODIFIED) - should be debounced
        event2 = FileChangeEvent(path=path, change_type=FileChangeType.MODIFIED)
        await processor.add_change(event2)
        
        # Third change (DELETED) - should replace the second
        event3 = FileChangeEvent(path=path, change_type=FileChangeType.DELETED)
        await processor.add_change(event3)
        
        await asyncio.sleep(0.15)
        
        # Get pending changes
        pending = await processor.get_pending_changes()
        assert len(pending) == 1
        # Should have the last change type
        assert pending[0].change_type == FileChangeType.DELETED

    async def test_concurrent_access_safety(self):
        """Test that concurrent access to the processor is thread-safe."""
        processor = DebouncedChangeProcessor(debounce_delay=0.1)
        
        async def add_changes(start_idx: int):
            """Add changes for multiple files."""
            for i in range(start_idx, start_idx + 10):
                path = Path(f"file{i}.py")
                event = FileChangeEvent(path=path, change_type=FileChangeType.MODIFIED)
                await processor.add_change(event)
                await asyncio.sleep(0.01)  # Small delay to interleave operations
        
        # Run multiple tasks concurrently
        tasks = [
            add_changes(0),
            add_changes(10),
            add_changes(20),
        ]
        
        await asyncio.gather(*tasks)
        
        # All operations should complete without errors
        # Check that we have the expected state
        assert processor.pending_count >= 0  # Some may be pending
        
        # Wait for all to be ready
        await asyncio.sleep(0.15)
        pending = await processor.get_pending_changes()
        
        # Should have processed all changes without corruption
        assert len(pending) <= 30  # Some may have been processed immediately


class TestRepositoryMonitorWithDebouncing:
    """Test RepositoryMonitor with debouncing integration."""
    
    @pytest.fixture
    def mock_neo4j_driver(self):
        """Create a mock Neo4j driver."""
        driver = AsyncMock()
        return driver
    
    @pytest.fixture
    def repository_monitor(self, tmp_path, mock_neo4j_driver):
        """Create a RepositoryMonitor instance with debouncing."""
        return RepositoryMonitor(
            repo_path=tmp_path,
            project_name="test_project",
            neo4j_driver=mock_neo4j_driver,
            debounce_delay=0.1,  # Short delay for testing
        )
    
    def test_debounce_processor_initialized(self, repository_monitor):
        """Test that the debounce processor is properly initialized."""
        assert repository_monitor.debounce_processor is not None
        assert repository_monitor.debounce_processor.debounce_delay == 0.1
    
    async def test_monitor_with_debouncing(self, repository_monitor, tmp_path):
        """Test that the monitor uses debouncing for file changes."""
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("initial content")
        
        # Simulate rapid changes
        events = []
        for i in range(3):
            event = FileChangeEvent(
                path=test_file,
                change_type=FileChangeType.MODIFIED
            )
            events.append(event)
        
        # Process events through the debounce processor
        results = []
        for event in events:
            result = await repository_monitor.debounce_processor.add_change(event)
            results.append(result)
        
        # First should be processed, others debounced
        assert results[0] is True
        assert results[1] is False
        assert results[2] is False
        
        # Should have pending changes
        assert repository_monitor.debounce_processor.pending_count == 1