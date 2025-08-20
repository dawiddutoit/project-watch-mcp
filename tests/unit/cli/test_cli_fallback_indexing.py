"""Test that full indexing fallback runs in the background (non-blocking)."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from project_watch_mcp.cli import main


@pytest.mark.asyncio
async def test_full_indexing_fallback_is_non_blocking():
    """Test that when falling back to full indexing for new repositories, it's non-blocking."""
    
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
        
        # Create mock file info for new repository
        mock_file_info = Mock()
        mock_file_info.path = Mock(spec=Path)
        mock_file_info.path.read_text.return_value = "test content"
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
        
        # Track if background task was created
        background_tasks_created = []
        original_create_task = asyncio.create_task
        
        def track_create_task(coro):
            task = original_create_task(coro)
            background_tasks_created.append(task)
            return task
        
        mock_server = AsyncMock()
        # Make the server run briefly then exit
        async def server_run_briefly():
            # Wait a bit to allow background task to be created
            await asyncio.sleep(0.1)
            # Then exit
            raise asyncio.CancelledError()
        
        mock_server.run_stdio_async = AsyncMock(side_effect=server_run_briefly)
        mock_create_mcp_server.return_value = mock_server
        
        # Test with skip_initial_index=False (default) for new repository
        with patch('asyncio.create_task', side_effect=track_create_task):
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
                pass  # Expected when server exits
        
        # Verify that a background task was created for indexing
        assert len(background_tasks_created) > 0, "No background tasks were created"
        
        # Find the background indexing task
        indexing_task = None
        for task in background_tasks_created:
            if task.get_name() == "background-indexing":
                indexing_task = task
                break
        
        assert indexing_task is not None, "Background indexing task was not created"
        
        # Verify that the server started immediately (non-blocking)
        mock_server.run_stdio_async.assert_called_once()
        
        # Verify that the appropriate log message was shown
        mock_logger.info.assert_any_call("Starting repository index in background (non-blocking)...")
        mock_logger.info.assert_any_call("✓ Background indexing task started")
        
        # Cancel the background task to clean up
        if not indexing_task.done():
            indexing_task.cancel()
            try:
                await indexing_task
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_background_indexing_handles_errors_gracefully():
    """Test that background indexing errors don't crash the server."""
    
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
        # Make scan_repository fail
        mock_monitor.scan_repository = AsyncMock(side_effect=Exception("Scan failed"))
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
        mock_neo4j_rag.return_value = mock_rag
        
        mock_server = AsyncMock()
        # Make the server run briefly then exit
        async def server_run_briefly():
            await asyncio.sleep(0.1)
            raise asyncio.CancelledError()
        
        mock_server.run_stdio_async = AsyncMock(side_effect=server_run_briefly)
        mock_create_mcp_server.return_value = mock_server
        
        # Test that server starts even if background indexing fails
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
            pass  # Expected when server exits
        
        # Verify that the server started despite the error
        mock_server.run_stdio_async.assert_called_once()
        
        # Verify that the error was logged
        mock_logger.error.assert_any_call("Background indexing failed: Scan failed")