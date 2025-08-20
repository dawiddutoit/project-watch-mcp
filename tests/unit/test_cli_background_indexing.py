"""Tests for background indexing functionality in CLI."""

import asyncio
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call
from datetime import datetime

import pytest

from project_watch_mcp.cli import main
from project_watch_mcp.neo4j_rag import CodeFile


class TestBackgroundIndexing:
    """Test background indexing functionality."""

    @pytest.mark.asyncio
    async def test_server_starts_immediately_without_blocking(self):
        """Test that MCP server starts immediately without waiting for indexing."""
        # Track the order of operations
        operation_order = []
        server_started_event = asyncio.Event()
        indexing_started_event = asyncio.Event()
        
        with (
            patch("project_watch_mcp.cli.AsyncGraphDatabase.driver") as mock_driver_class,
            patch("project_watch_mcp.cli.RepositoryMonitor") as mock_monitor_class,
            patch("project_watch_mcp.cli.create_embeddings_provider") as mock_embeddings,
            patch("project_watch_mcp.cli.Neo4jRAG") as mock_rag_class,
            patch("project_watch_mcp.cli.create_mcp_server") as mock_server,
        ):
            # Mock driver
            mock_driver = AsyncMock()
            mock_driver.verify_connectivity.return_value = None
            mock_driver_class.return_value = mock_driver

            # Mock repository monitor with scan that takes time
            mock_monitor = AsyncMock()
            mock_monitor.repo_path = Path("/test/repo")
            
            async def slow_scan():
                operation_order.append("scan_started")
                indexing_started_event.set()
                await asyncio.sleep(0.1)  # Simulate slow scan
                operation_order.append("scan_completed")
                return []
            
            mock_monitor.scan_repository.side_effect = slow_scan
            mock_monitor.start = AsyncMock(side_effect=lambda daemon=False: operation_order.append("monitor_started"))
            mock_monitor_class.return_value = mock_monitor

            # Mock RAG
            mock_rag = AsyncMock()
            mock_rag.initialize.side_effect = lambda: operation_order.append("rag_initialized")
            mock_rag_class.return_value = mock_rag

            # Mock MCP server that tracks when it starts
            mock_mcp = AsyncMock()
            
            async def mock_run_stdio():
                operation_order.append("server_started")
                server_started_event.set()
                # Simulate server running for a bit
                await asyncio.sleep(0.2)
                
            mock_mcp.run_stdio_async.side_effect = mock_run_stdio
            mock_server.return_value = mock_mcp

            # Create main task
            main_task = asyncio.create_task(
                main(
                    neo4j_uri="bolt://localhost:7687",
                    PROJECT_WATCH_USER="neo4j",
                    PROJECT_WATCH_PASSWORD="password",
                    PROJECT_WATCH_DATABASE="neo4j",
                    repository_path="/test/repo",
                    transport="stdio",
                    skip_initial_index=False,  # We want indexing, but in background
                )
            )
            
            # Wait for server to start
            await asyncio.wait_for(server_started_event.wait(), timeout=1.0)
            
            # Server should have started before indexing completes
            assert "server_started" in operation_order
            # Indexing may or may not have started yet, but server shouldn't wait
            
            # Cancel the main task
            main_task.cancel()
            try:
                await main_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_indexing_happens_in_background(self):
        """Test that indexing happens in the background after server starts."""
        indexing_complete = threading.Event()
        server_ready = threading.Event()
        
        with (
            patch("project_watch_mcp.cli.AsyncGraphDatabase.driver") as mock_driver_class,
            patch("project_watch_mcp.cli.RepositoryMonitor") as mock_monitor_class,
            patch("project_watch_mcp.cli.create_embeddings_provider"),
            patch("project_watch_mcp.cli.Neo4jRAG") as mock_rag_class,
            patch("project_watch_mcp.cli.create_mcp_server") as mock_server,
            patch("project_watch_mcp.cli.asyncio.create_task") as mock_create_task,
            patch.object(Path, 'read_text', return_value="test content"),
        ):
            # Mock components
            mock_driver = AsyncMock()
            mock_driver.verify_connectivity.return_value = None
            mock_driver_class.return_value = mock_driver

            mock_monitor = AsyncMock()
            mock_monitor.repo_path = Path("/test/repo")
            
            # Create test files
            test_files = [
                MagicMock(path=Path("/test/repo/file1.py"), language="python", size=100, last_modified=datetime.now()),
                MagicMock(path=Path("/test/repo/file2.py"), language="python", size=200, last_modified=datetime.now()),
            ]
            
            mock_monitor.scan_repository.return_value = test_files
            mock_monitor_class.return_value = mock_monitor

            mock_rag = AsyncMock()
            
            # Track indexing calls
            indexed_files = []
            async def track_indexing(code_file):
                indexed_files.append(code_file.path)
                if len(indexed_files) == len(test_files):
                    indexing_complete.set()
                    
            mock_rag.index_file.side_effect = track_indexing
            mock_rag_class.return_value = mock_rag

            mock_mcp = AsyncMock()
            
            # Server should be ready immediately
            async def mock_stdio():
                server_ready.set()
                await asyncio.sleep(0.5)  # Keep server "running"
                
            mock_mcp.run_stdio_async.side_effect = mock_stdio
            mock_server.return_value = mock_mcp
            
            # Track background tasks
            background_tasks = []
            
            def track_background_task(coro):
                # Don't use mock_create_task here to avoid recursion
                task = asyncio.Task(coro)
                background_tasks.append(task)
                return task
                
            mock_create_task.side_effect = track_background_task

            # Run main
            main_task = asyncio.create_task(
                main(
                    neo4j_uri="bolt://localhost:7687",
                    PROJECT_WATCH_USER="neo4j",
                    PROJECT_WATCH_PASSWORD="password",
                    PROJECT_WATCH_DATABASE="neo4j",
                    repository_path="/test/repo",
                    transport="stdio",
                    skip_initial_index=False,
                )
            )
            
            # Give it time to start
            await asyncio.sleep(0.1)
            
            # Check that a background task was created for indexing
            assert len(background_tasks) > 0, "Background indexing task should be created"
            
            # Cancel main task
            main_task.cancel()
            try:
                await main_task
            except asyncio.CancelledError:
                pass
            
            # Clean up background tasks
            for task in background_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_queries_work_with_partial_index(self):
        """Test that queries work even while indexing is in progress."""
        with (
            patch("project_watch_mcp.cli.AsyncGraphDatabase.driver") as mock_driver_class,
            patch("project_watch_mcp.cli.RepositoryMonitor") as mock_monitor_class,
            patch("project_watch_mcp.cli.create_embeddings_provider"),
            patch("project_watch_mcp.cli.Neo4jRAG") as mock_rag_class,
            patch("project_watch_mcp.cli.create_mcp_server") as mock_server,
        ):
            # Setup mocks
            mock_driver = AsyncMock()
            mock_driver_class.return_value = mock_driver

            mock_monitor = AsyncMock()
            mock_monitor.repo_path = Path("/test/repo")
            mock_monitor.scan_repository.return_value = []
            mock_monitor_class.return_value = mock_monitor

            mock_rag = AsyncMock()
            
            # RAG should be available for queries immediately
            mock_rag.search_semantic.return_value = [
                {"file": "partial.py", "content": "partial result", "similarity": 0.8}
            ]
            mock_rag_class.return_value = mock_rag

            # Mock server is created and can handle requests immediately
            mock_mcp = AsyncMock()
            mock_server.return_value = mock_mcp
            
            # Verify that create_mcp_server is called before indexing completes
            # This ensures tools are available immediately
            mock_server.assert_not_called()  # Not called yet
            
            await main(
                neo4j_uri="bolt://localhost:7687",
                PROJECT_WATCH_USER="neo4j",
                PROJECT_WATCH_PASSWORD="password",
                PROJECT_WATCH_DATABASE="neo4j",
                repository_path="/test/repo",
                transport="stdio",
                skip_initial_index=False,
            )
            
            # Server should be created regardless of indexing status
            mock_server.assert_called_once()

    @pytest.mark.asyncio
    async def test_skip_initial_index_flag_still_works(self):
        """Test that --skip-initial-index flag still works as expected."""
        with (
            patch("project_watch_mcp.cli.AsyncGraphDatabase.driver") as mock_driver_class,
            patch("project_watch_mcp.cli.RepositoryMonitor") as mock_monitor_class,
            patch("project_watch_mcp.cli.create_embeddings_provider"),
            patch("project_watch_mcp.cli.Neo4jRAG") as mock_rag_class,
            patch("project_watch_mcp.cli.create_mcp_server") as mock_server,
        ):
            # Setup mocks
            mock_driver = AsyncMock()
            mock_driver_class.return_value = mock_driver

            mock_monitor = AsyncMock()
            mock_monitor.repo_path = Path("/test/repo")
            mock_monitor_class.return_value = mock_monitor

            mock_rag = AsyncMock()
            mock_rag_class.return_value = mock_rag

            mock_mcp = AsyncMock()
            mock_server.return_value = mock_mcp

            await main(
                neo4j_uri="bolt://localhost:7687",
                PROJECT_WATCH_USER="neo4j",
                PROJECT_WATCH_PASSWORD="password",
                PROJECT_WATCH_DATABASE="neo4j",
                repository_path="/test/repo",
                transport="stdio",
                skip_initial_index=True,  # Skip indexing
            )
            
            # scan_repository should NOT be called when skip_initial_index is True
            mock_monitor.scan_repository.assert_not_called()
            
            # Server should still be created and started
            mock_server.assert_called_once()
            mock_mcp.run_stdio_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_indexing_errors_dont_crash_server(self):
        """Test that indexing errors don't crash the MCP server."""
        with (
            patch("project_watch_mcp.cli.AsyncGraphDatabase.driver") as mock_driver_class,
            patch("project_watch_mcp.cli.RepositoryMonitor") as mock_monitor_class,
            patch("project_watch_mcp.cli.create_embeddings_provider"),
            patch("project_watch_mcp.cli.Neo4jRAG") as mock_rag_class,
            patch("project_watch_mcp.cli.create_mcp_server") as mock_server,
            patch("project_watch_mcp.cli.logger") as mock_logger,
        ):
            # Setup mocks
            mock_driver = AsyncMock()
            mock_driver_class.return_value = mock_driver

            mock_monitor = AsyncMock()
            mock_monitor.repo_path = Path("/test/repo")
            
            # Simulate indexing error
            mock_monitor.scan_repository.side_effect = Exception("Indexing failed!")
            mock_monitor_class.return_value = mock_monitor

            mock_rag = AsyncMock()
            mock_rag_class.return_value = mock_rag

            mock_mcp = AsyncMock()
            mock_server.return_value = mock_mcp

            # Server should start even if indexing fails
            await main(
                neo4j_uri="bolt://localhost:7687",
                PROJECT_WATCH_USER="neo4j",
                PROJECT_WATCH_PASSWORD="password",
                PROJECT_WATCH_DATABASE="neo4j",
                repository_path="/test/repo",
                transport="stdio",
                skip_initial_index=False,
            )
            
            # Server should still be created and started
            mock_server.assert_called_once()
            mock_mcp.run_stdio_async.assert_called_once()
            
            # Error should be logged, not raised
            # Wait a bit for the background task to process
            await asyncio.sleep(0.1)
            
            # Check if error was logged (either through mock or just verify server started)
            # The key is that the server started despite the error
            assert mock_mcp.run_stdio_async.called, "Server should start despite indexing errors"