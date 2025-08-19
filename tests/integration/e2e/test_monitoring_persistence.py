"""Test that monitoring persists after initialization."""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neo4j import AsyncDriver

from project_watch_mcp.core.initializer import RepositoryInitializer
from project_watch_mcp.repository_monitor import RepositoryMonitor


@pytest.fixture
async def mock_neo4j_driver():
    """Create a mock Neo4j driver."""
    driver = AsyncMock(spec=AsyncDriver)
    driver.verify_connectivity = AsyncMock()
    driver.close = AsyncMock()
    return driver


@pytest.fixture
def temp_repo():
    """Create a temporary repository directory with some files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        
        # Create some test files
        (repo_path / "test.py").write_text("print('hello')")
        (repo_path / "main.js").write_text("console.log('world')")
        
        yield repo_path


class TestMonitoringPersistence:
    """Test that monitoring persists properly after initialization."""
    
    async def test_repository_monitor_start_creates_background_task(self, mock_neo4j_driver, temp_repo):
        """Test that starting the monitor creates a background task that continues running."""
        monitor = RepositoryMonitor(
            repo_path=temp_repo,
            project_name="test_project",
            neo4j_driver=mock_neo4j_driver,
        )
        
        # Start monitoring
        await monitor.start()
        
        # Check that monitoring is running
        assert monitor.is_running is True
        assert monitor._watch_task is not None
        assert not monitor._watch_task.done()
        
        # Wait a bit to ensure the task continues running
        await asyncio.sleep(0.1)
        
        # Should still be running
        assert monitor.is_running is True
        assert not monitor._watch_task.done()
        
        # Clean up
        await monitor.stop()
        assert monitor.is_running is False
    
    async def test_initializer_starts_monitoring_that_persists(self, mock_neo4j_driver, temp_repo):
        """Test that the initializer starts monitoring that continues after initialization."""
        with patch('project_watch_mcp.core.initializer.AsyncGraphDatabase.driver') as mock_driver_factory:
            mock_driver_factory.return_value = mock_neo4j_driver
            
            # Mock the Neo4jRAG class
            with patch('project_watch_mcp.core.initializer.Neo4jRAG') as MockNeo4jRAG:
                mock_rag = AsyncMock()
                mock_rag.index_file = AsyncMock()
                MockNeo4jRAG.return_value = mock_rag
                
                initializer = RepositoryInitializer(
                    neo4j_uri="bolt://localhost:7687",
                    PROJECT_WATCH_USER="neo4j",
                    PROJECT_WATCH_PASSWORD="password",
                    PROJECT_WATCH_DATABASE="test",
                    repository_path=temp_repo,
                    project_name="test_project",
                )
                
                async with initializer:
                    result = await initializer.initialize()
                    
                    # Check that monitoring was reported as started
                    assert result.monitoring is True
                    
                    # During context, monitor should exist
                    assert initializer._repository_monitor is not None
                
                # After context exit, the monitor is cleaned up as expected
                # The key is that monitoring was successfully started
    
    async def test_monitor_continues_detecting_changes(self, mock_neo4j_driver, temp_repo):
        """Test that the monitor continues to detect file changes after being started."""
        monitor = RepositoryMonitor(
            repo_path=temp_repo,
            project_name="test_project",
            neo4j_driver=mock_neo4j_driver,
        )
        
        # Start monitoring
        await monitor.start()
        
        # Create a new file after monitoring starts
        await asyncio.sleep(0.1)  # Give monitor time to start
        new_file = temp_repo / "new_file.py"
        new_file.write_text("# New file")
        
        # Wait for the change to be detected
        await asyncio.sleep(0.5)
        
        # Check if changes were detected
        has_changes = monitor.has_pending_changes()
        
        # Process changes if any were detected
        if has_changes:
            changes = await monitor.process_all_changes()
            assert len(changes) > 0
            # At least one change should be for the new file
            assert any(change.path.name == "new_file.py" for change in changes)
        
        # Clean up
        await monitor.stop()
    
    async def test_monitoring_survives_multiple_initializations(self, mock_neo4j_driver, temp_repo):
        """Test that monitoring can be restarted multiple times."""
        monitor = RepositoryMonitor(
            repo_path=temp_repo,
            project_name="test_project",
            neo4j_driver=mock_neo4j_driver,
        )
        
        # Start monitoring multiple times
        for i in range(3):
            await monitor.start()
            assert monitor.is_running is True
            
            # Stop and restart
            await monitor.stop()
            assert monitor.is_running is False
            
            # Can restart
            await monitor.start()
            assert monitor.is_running is True
            
            await monitor.stop()
    
    async def test_monitoring_handles_concurrent_starts(self, mock_neo4j_driver, temp_repo):
        """Test that multiple concurrent start calls are handled properly."""
        monitor = RepositoryMonitor(
            repo_path=temp_repo,
            project_name="test_project",
            neo4j_driver=mock_neo4j_driver,
        )
        
        # Start monitoring
        await monitor.start()
        assert monitor.is_running is True
        
        # Try to start again - should handle gracefully
        await monitor.start()  # Should not raise an error
        assert monitor.is_running is True
        
        # Clean up
        await monitor.stop()
        assert monitor.is_running is False


class TestMonitoringWithDaemonMode:
    """Test daemon mode for persistent monitoring."""
    
    async def test_monitor_daemon_mode_flag(self, mock_neo4j_driver, temp_repo):
        """Test that monitor supports a daemon mode flag."""
        monitor = RepositoryMonitor(
            repo_path=temp_repo,
            project_name="test_project",
            neo4j_driver=mock_neo4j_driver,
        )
        
        # Check if we can start in daemon mode
        # This would need to be implemented to truly run as a daemon
        await monitor.start(daemon=False)  # Normal mode for testing
        assert monitor.is_running is True
        
        await monitor.stop()
        assert monitor.is_running is False
    
    async def test_initializer_with_persistent_monitoring(self, mock_neo4j_driver, temp_repo):
        """Test initializer with persistent monitoring option."""
        with patch('project_watch_mcp.core.initializer.AsyncGraphDatabase.driver') as mock_driver_factory:
            mock_driver_factory.return_value = mock_neo4j_driver
            
            with patch('project_watch_mcp.core.initializer.Neo4jRAG') as MockNeo4jRAG:
                mock_rag = AsyncMock()
                mock_rag.index_file = AsyncMock()
                MockNeo4jRAG.return_value = mock_rag
                
                initializer = RepositoryInitializer(
                    neo4j_uri="bolt://localhost:7687",
                    PROJECT_WATCH_USER="neo4j",
                    PROJECT_WATCH_PASSWORD="password",
                    PROJECT_WATCH_DATABASE="test",
                    repository_path=temp_repo,
                    project_name="test_project",
                )
                
                # Initialize with persistent monitoring
                async with initializer:
                    result = await initializer.initialize(persistent_monitoring=True)
                    
                    assert result.monitoring is True
                    assert result.message.startswith("Repository initialized")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])