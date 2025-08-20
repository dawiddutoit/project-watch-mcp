"""Test for the --skip-initial-index CLI flag."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, MagicMock, patch

import pytest

from project_watch_mcp.cli import main


@pytest.mark.asyncio
async def test_main_with_skip_initial_index():
    """Test that main function skips initial indexing when skip_initial_index=True."""
    
    # Mock all the dependencies
    with patch('project_watch_mcp.cli.AsyncGraphDatabase') as mock_graph_db, \
         patch('project_watch_mcp.cli.ProjectConfig') as mock_project_config, \
         patch('project_watch_mcp.cli.RepositoryMonitor') as mock_repo_monitor, \
         patch('project_watch_mcp.cli.EmbeddingConfig') as mock_embed_config, \
         patch('project_watch_mcp.cli.create_embeddings_provider') as mock_create_embeddings, \
         patch('project_watch_mcp.cli.Neo4jRAG') as mock_neo4j_rag, \
         patch('project_watch_mcp.cli.create_mcp_server') as mock_create_mcp_server, \
         patch('project_watch_mcp.cli.logger') as mock_logger:
        
        # Setup mocks
        mock_driver = AsyncMock()
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.verify_connectivity = AsyncMock()
        mock_driver.close = AsyncMock()
        
        mock_project = Mock()
        mock_project.name = "test-project"
        mock_project_config.from_repository_path.return_value = mock_project
        
        mock_monitor = AsyncMock()
        mock_monitor.scan_repository = AsyncMock(return_value=[])
        mock_monitor.start = AsyncMock()
        mock_monitor.stop = AsyncMock()
        mock_repo_monitor.return_value = mock_monitor
        
        mock_embed_config.from_env.return_value = Mock(
            provider="disabled",
            api_key=None,
            model=None,
            dimension=None
        )
        mock_create_embeddings.return_value = None
        
        mock_rag = AsyncMock()
        mock_rag.initialize = AsyncMock()
        mock_rag.index_file = AsyncMock()
        mock_neo4j_rag.return_value = mock_rag
        
        mock_server = AsyncMock()
        mock_server.run_stdio_async = AsyncMock(side_effect=asyncio.CancelledError())
        mock_create_mcp_server.return_value = mock_server
        
        # Test with skip_initial_index=True
        try:
            await main(
                neo4j_uri="bolt://localhost:7687",
                PROJECT_WATCH_USER="neo4j",
                PROJECT_WATCH_PASSWORD="password",
                PROJECT_WATCH_DATABASE="neo4j",
                repository_path="/test/repo",
                project_name="test-project",
                transport="stdio",
                host="127.0.0.1",
                port=8000,
                path="/mcp/",
                file_patterns="*.py",
                skip_initial_index=True
            )
        except asyncio.CancelledError:
            pass  # Expected when we cancel the server
        
        # Verify that scan_repository was NOT called when skip_initial_index=True
        mock_monitor.scan_repository.assert_not_called()
        
        # Verify that the appropriate log message was shown
        mock_logger.info.assert_any_call("⚠ Skipping initial repository indexing (--skip-initial-index flag set)")


@pytest.mark.asyncio
async def test_main_without_skip_initial_index():
    """Test that main function performs initial indexing when skip_initial_index=False."""
    
    # Mock all the dependencies
    with patch('project_watch_mcp.cli.AsyncGraphDatabase') as mock_graph_db, \
         patch('project_watch_mcp.cli.ProjectConfig') as mock_project_config, \
         patch('project_watch_mcp.cli.RepositoryMonitor') as mock_repo_monitor, \
         patch('project_watch_mcp.cli.EmbeddingConfig') as mock_embed_config, \
         patch('project_watch_mcp.cli.create_embeddings_provider') as mock_create_embeddings, \
         patch('project_watch_mcp.cli.Neo4jRAG') as mock_neo4j_rag, \
         patch('project_watch_mcp.cli.create_mcp_server') as mock_create_mcp_server, \
         patch('project_watch_mcp.cli.logger') as mock_logger:
        
        # Setup mocks
        mock_driver = AsyncMock()
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.verify_connectivity = AsyncMock()
        mock_driver.close = AsyncMock()
        
        mock_project = Mock()
        mock_project.name = "test-project"
        mock_project_config.from_repository_path.return_value = mock_project
        
        # Create a mock file info with proper path mocking
        mock_file_info = Mock()
        mock_path = Mock()
        mock_path.read_text = Mock(return_value="test content")
        mock_path.__str__ = Mock(return_value="/test/file.py")
        mock_file_info.path = mock_path
        mock_file_info.language = "python"
        mock_file_info.size = 100
        mock_file_info.last_modified = 123456789
        
        mock_monitor = AsyncMock()
        mock_monitor.scan_repository = AsyncMock(return_value=[mock_file_info])
        mock_monitor.start = AsyncMock()
        mock_monitor.stop = AsyncMock()
        mock_repo_monitor.return_value = mock_monitor
        
        mock_embed_config.from_env.return_value = Mock(
            provider="disabled",
            api_key=None,
            model=None,
            dimension=None
        )
        mock_create_embeddings.return_value = None
        
        mock_rag = AsyncMock()
        mock_rag.initialize = AsyncMock()
        mock_rag.index_file = AsyncMock()
        mock_neo4j_rag.return_value = mock_rag
        
        mock_server = AsyncMock()
        mock_server.run_stdio_async = AsyncMock(side_effect=asyncio.CancelledError())
        mock_create_mcp_server.return_value = mock_server
        
        # Test with skip_initial_index=False (default)
        try:
            await main(
                neo4j_uri="bolt://localhost:7687",
                PROJECT_WATCH_USER="neo4j",
                PROJECT_WATCH_PASSWORD="password",
                PROJECT_WATCH_DATABASE="neo4j",
                repository_path="/test/repo",
                project_name="test-project",
                transport="stdio",
                host="127.0.0.1",
                port=8000,
                path="/mcp/",
                file_patterns="*.py",
                skip_initial_index=False
            )
        except asyncio.CancelledError:
            pass  # Expected when we cancel the server
        
        # Give the background task a moment to start and call scan_repository
        await asyncio.sleep(0.2)
        
        # Verify that scan_repository WAS called when skip_initial_index=False
        mock_monitor.scan_repository.assert_called_once()
        
        # Since indexing is async, we may need to wait longer or just verify the task was created
        # For unit test, it's enough to verify the background task was started
        mock_logger.info.assert_any_call("Starting repository index in background (non-blocking)...")
        mock_logger.info.assert_any_call("✓ Background indexing task started")