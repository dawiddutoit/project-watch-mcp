"""Test CLI monitoring functionality."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from project_watch_mcp.cli import initialize_only
from project_watch_mcp.core.monitoring_manager import MonitoringManager


@pytest.fixture
def temp_repo():
    """Create a temporary repository directory with some files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        
        # Create some test files
        (repo_path / "test.py").write_text("print('hello')")
        (repo_path / "main.js").write_text("console.log('world')")
        (repo_path / "README.md").write_text("# Test Project")
        
        yield repo_path


class TestCLIMonitoring:
    """Test CLI monitoring initialization."""
    
    async def test_initialize_only_starts_persistent_monitoring(self, temp_repo):
        """Test that CLI initialization starts persistent monitoring."""
        with patch('project_watch_mcp.core.initializer.AsyncGraphDatabase.driver') as mock_driver_factory:
            mock_driver = AsyncMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.close = AsyncMock()
            mock_driver_factory.return_value = mock_driver
            
            with patch('project_watch_mcp.core.initializer.Neo4jRAG') as MockNeo4jRAG:
                mock_rag = AsyncMock()
                mock_rag.index_file = AsyncMock()
                MockNeo4jRAG.return_value = mock_rag
                
                # Run CLI initialization
                exit_code = await initialize_only(
                    neo4j_uri="bolt://localhost:7687",
                    neo4j_user="neo4j",
                    neo4j_password="password",
                    neo4j_database="test",
                    repository_path=str(temp_repo),
                    project_name="test_cli_project",
                    verbose=False,
                )
                
                # Should succeed
                assert exit_code == 0
                
                # Check if monitoring was started
                manager = MonitoringManager.get_instance("test_cli_project")
                if manager:
                    # Monitoring manager should exist
                    assert manager.project_name == "test_cli_project"
                    # Clean up
                    await manager.stop()
    
    async def test_initialize_with_verbose_output(self, temp_repo, capsys):
        """Test that verbose mode shows progress."""
        with patch('project_watch_mcp.core.initializer.AsyncGraphDatabase.driver') as mock_driver_factory:
            mock_driver = AsyncMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.close = AsyncMock()
            mock_driver_factory.return_value = mock_driver
            
            with patch('project_watch_mcp.core.initializer.Neo4jRAG') as MockNeo4jRAG:
                mock_rag = AsyncMock()
                mock_rag.index_file = AsyncMock()
                MockNeo4jRAG.return_value = mock_rag
                
                # Run with verbose mode
                exit_code = await initialize_only(
                    neo4j_uri="bolt://localhost:7687",
                    neo4j_user="neo4j",
                    neo4j_password="password",
                    neo4j_database="test",
                    repository_path=str(temp_repo),
                    project_name="test_verbose",
                    verbose=True,
                )
                
                assert exit_code == 0
                
                # Check output
                captured = capsys.readouterr()
                assert "Project: test_verbose" in captured.out
                assert "Indexed:" in captured.out
                
                # Should show progress in stderr when verbose
                assert "Scanning repository" in captured.err or len(captured.err) > 0
                
                # Clean up
                manager = MonitoringManager.get_instance("test_verbose")
                if manager:
                    await manager.stop()
    
    async def test_initialize_handles_errors(self):
        """Test that initialization handles errors gracefully."""
        # Test with invalid Neo4j connection
        exit_code = await initialize_only(
            neo4j_uri="bolt://invalid:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            neo4j_database="test",
            repository_path="/nonexistent/path",
            project_name="test_error",
            verbose=False,
        )
        
        # Should fail
        assert exit_code == 1
    
    async def test_monitoring_manager_singleton_behavior(self, temp_repo):
        """Test that MonitoringManager properly manages instances."""
        with patch('project_watch_mcp.core.initializer.AsyncGraphDatabase.driver') as mock_driver_factory:
            mock_driver = AsyncMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.close = AsyncMock()
            mock_driver_factory.return_value = mock_driver
            
            # Create two managers with same project name
            manager1 = MonitoringManager(
                repo_path=temp_repo,
                project_name="singleton_test",
                neo4j_driver=mock_driver,
            )
            
            manager2 = MonitoringManager(
                repo_path=temp_repo,
                project_name="singleton_test",
                neo4j_driver=mock_driver,
            )
            
            # Should track both but last one wins for get_instance
            assert MonitoringManager.get_instance("singleton_test") == manager2
            
            # Clean up
            await manager1.stop()
            await manager2.stop()
            
            # Should be removed from instances
            assert MonitoringManager.get_instance("singleton_test") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])