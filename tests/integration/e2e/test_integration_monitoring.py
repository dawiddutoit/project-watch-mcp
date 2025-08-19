"""Integration tests for monitoring functionality."""

import asyncio
import tempfile
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, patch

from project_watch_mcp.core import RepositoryInitializer, MonitoringManager


@pytest.fixture
def temp_repo():
    """Create a temporary repository directory with some files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        
        # Create directory structure
        (repo_path / "src").mkdir()
        (repo_path / "tests").mkdir()
        
        # Create some test files
        (repo_path / "src" / "main.py").write_text("""
def main():
    print("Hello World")
    
if __name__ == "__main__":
    main()
""")
        (repo_path / "src" / "utils.py").write_text("""
def helper():
    return 42
""")
        (repo_path / "tests" / "test_main.py").write_text("""
import pytest

def test_main():
    assert True
""")
        (repo_path / "README.md").write_text("# Test Project\n\nThis is a test project.")
        (repo_path / ".gitignore").write_text("*.pyc\n__pycache__/\n.venv/")
        
        yield repo_path


class TestMonitoringIntegration:
    """Integration tests for monitoring system."""
    
    async def test_full_initialization_and_monitoring_flow(self, temp_repo):
        """Test the complete flow from initialization to monitoring."""
        with patch('project_watch_mcp.core.initializer.AsyncGraphDatabase.driver') as mock_driver_factory:
            mock_driver = AsyncMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.close = AsyncMock()
            mock_driver_factory.return_value = mock_driver
            
            with patch('project_watch_mcp.core.initializer.Neo4jRAG') as MockNeo4jRAG:
                mock_rag = AsyncMock()
                mock_rag.index_file = AsyncMock()
                MockNeo4jRAG.return_value = mock_rag
                
                # Initialize repository with persistent monitoring
                initializer = RepositoryInitializer(
                    neo4j_uri="bolt://localhost:7687",
                    PROJECT_WATCH_USER="neo4j",
                    PROJECT_WATCH_PASSWORD="password",
                    PROJECT_WATCH_DATABASE="test",
                    repository_path=temp_repo,
                    project_name="integration_test",
                )
                
                async with initializer:
                    result = await initializer.initialize(persistent_monitoring=True)
                
                # Verify initialization results
                assert result.indexed > 0
                assert result.total >= result.indexed
                assert result.monitoring is True
                
                # Check that monitoring manager was created
                manager = MonitoringManager.get_instance("integration_test")
                assert manager is not None
                assert manager.project_name == "integration_test"
                assert manager.is_running()
                
                # Simulate file change
                new_file = temp_repo / "src" / "new_module.py"
                new_file.write_text("# New module\nprint('new')")
                
                # Give monitor time to detect change
                await asyncio.sleep(0.5)
                
                # Check if monitor is still running
                assert manager.is_running()
                
                # Clean up
                await manager.stop()
                assert not manager.is_running()
                assert MonitoringManager.get_instance("integration_test") is None
    
    async def test_multiple_projects_monitoring(self, temp_repo):
        """Test monitoring multiple projects simultaneously."""
        with patch('project_watch_mcp.core.initializer.AsyncGraphDatabase.driver') as mock_driver_factory:
            mock_driver = AsyncMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.close = AsyncMock()
            mock_driver_factory.return_value = mock_driver
            
            # Create managers for multiple projects
            manager1 = MonitoringManager(
                repo_path=temp_repo,
                project_name="project1",
                neo4j_driver=mock_driver,
            )
            
            manager2 = MonitoringManager(
                repo_path=temp_repo,
                project_name="project2",
                neo4j_driver=mock_driver,
            )
            
            # Start both with properly mocked monitor
            with patch('project_watch_mcp.core.monitoring_manager.RepositoryMonitor') as MockMonitor:
                mock_monitor = AsyncMock()
                mock_monitor.start = AsyncMock()
                mock_monitor.is_running = True
                MockMonitor.return_value = mock_monitor
                
                assert await manager1.start_persistent_monitoring()
                assert await manager2.start_persistent_monitoring()
            
            # Both should be tracked
            assert MonitoringManager.get_instance("project1") == manager1
            assert MonitoringManager.get_instance("project2") == manager2
            assert MonitoringManager.is_monitoring("project1")
            assert MonitoringManager.is_monitoring("project2")
            
            # Stop one
            await manager1.stop()
            assert MonitoringManager.get_instance("project1") is None
            assert MonitoringManager.get_instance("project2") == manager2
            
            # Stop the other
            await manager2.stop()
            assert MonitoringManager.get_instance("project2") is None
    
    async def test_monitoring_restart_after_failure(self, temp_repo):
        """Test that monitoring can be restarted after a failure."""
        with patch('project_watch_mcp.core.initializer.AsyncGraphDatabase.driver') as mock_driver_factory:
            mock_driver = AsyncMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.close = AsyncMock()
            mock_driver_factory.return_value = mock_driver
            
            manager = MonitoringManager(
                repo_path=temp_repo,
                project_name="restart_test",
                neo4j_driver=mock_driver,
            )
            
            # Start monitoring with properly mocked monitor
            with patch('project_watch_mcp.core.monitoring_manager.RepositoryMonitor') as MockMonitor:
                mock_monitor = AsyncMock()
                mock_monitor.start = AsyncMock()
                mock_monitor.is_running = True
                MockMonitor.return_value = mock_monitor
                
                assert await manager.start_persistent_monitoring()
                assert manager.is_running()
            
            # Simulate failure by stopping
            await manager.stop()
            assert not manager.is_running()
            
            # Should be able to restart
            with patch('project_watch_mcp.core.monitoring_manager.RepositoryMonitor') as MockMonitor:
                mock_monitor = AsyncMock()
                mock_monitor.start = AsyncMock()
                mock_monitor.is_running = True
                MockMonitor.return_value = mock_monitor
                
                manager = MonitoringManager(
                    repo_path=temp_repo,
                    project_name="restart_test",
                    neo4j_driver=mock_driver,
                )
                assert await manager.start_persistent_monitoring()
                assert manager.is_running()
            
            # Clean up
            await manager.stop()
    
    async def test_shutdown_all_managers(self, temp_repo):
        """Test shutting down all monitoring managers at once."""
        with patch('project_watch_mcp.core.initializer.AsyncGraphDatabase.driver') as mock_driver_factory:
            mock_driver = AsyncMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.close = AsyncMock()
            mock_driver_factory.return_value = mock_driver
            
            # Create multiple managers
            managers = []
            for i in range(3):
                manager = MonitoringManager(
                    repo_path=temp_repo,
                    project_name=f"shutdown_test_{i}",
                    neo4j_driver=mock_driver,
                )
                managers.append(manager)
                
                with patch('project_watch_mcp.core.monitoring_manager.RepositoryMonitor') as MockMonitor:
                    mock_monitor = AsyncMock()
                    mock_monitor.start = AsyncMock()
                    mock_monitor.is_running = True
                    MockMonitor.return_value = mock_monitor
                    
                    await manager.start_persistent_monitoring()
            
            # All should be running
            for i in range(3):
                assert MonitoringManager.is_monitoring(f"shutdown_test_{i}")
            
            # Shutdown all
            await MonitoringManager.shutdown_all()
            
            # None should be running
            for i in range(3):
                assert not MonitoringManager.is_monitoring(f"shutdown_test_{i}")
                assert MonitoringManager.get_instance(f"shutdown_test_{i}") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])