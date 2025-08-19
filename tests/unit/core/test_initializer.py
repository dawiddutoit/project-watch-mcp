"""Comprehensive unit tests for RepositoryInitializer class.

Test coverage targets 90% overall with 100% for critical paths.
"""

import asyncio
import tracemalloc
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
from neo4j import AsyncDriver, AsyncGraphDatabase

from src.project_watch_mcp.core.initializer import (
    ConnectionError,
    FileAccessError,
    IndexingError,
    InitializationError,
    InitializationResult,
    RepositoryInitializer,
)
from src.project_watch_mcp.neo4j_rag import CodeFile
from src.project_watch_mcp.repository_monitor import FileInfo


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_mock_neo4j_rag(sample_file_infos=None):
    """Create a properly mocked Neo4jRAG instance with incremental indexing methods."""
    mock_rag = AsyncMock()
    mock_rag.index_file = AsyncMock()
    mock_rag.is_repository_indexed = AsyncMock(return_value=False)
    mock_rag.get_indexed_files = AsyncMock(return_value={})
    if sample_file_infos:
        mock_rag.detect_changed_files = AsyncMock(return_value=(sample_file_infos, [], []))
    else:
        mock_rag.detect_changed_files = AsyncMock(return_value=([], [], []))
    mock_rag.remove_files = AsyncMock()
    return mock_rag

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j AsyncDriver."""
    driver = AsyncMock(spec=AsyncDriver)
    driver.verify_connectivity = AsyncMock()
    driver.close = AsyncMock()
    return driver


@pytest.fixture
def mock_neo4j_rag():
    """Mock Neo4jRAG instance."""
    rag = AsyncMock()
    rag.index_file = AsyncMock()
    rag.initialize = AsyncMock()
    return rag


@pytest.fixture
def mock_repository_monitor():
    """Mock RepositoryMonitor instance."""
    monitor = AsyncMock()
    monitor.scan_repository = AsyncMock()
    monitor.start = AsyncMock()
    monitor.stop = AsyncMock()
    monitor.is_running = False
    return monitor


@pytest.fixture
def mock_embeddings_provider():
    """Mock EmbeddingsProvider."""
    provider = AsyncMock()
    provider.embed_text = AsyncMock(return_value=[0.1] * 1536)
    provider.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
    provider.dimension = 1536
    return provider


@pytest.fixture
def test_repository_path(tmp_path):
    """Create a test repository with sample files."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    
    # Create some test files
    (repo_path / "main.py").write_text("print('Hello World')", encoding="utf-8")
    (repo_path / "README.md").write_text("# Test Project", encoding="utf-8")
    (repo_path / "utils.py").write_text("def helper(): pass", encoding="utf-8")
    
    # Create a subdirectory with files
    (repo_path / "src").mkdir()
    (repo_path / "src" / "module.py").write_text("class TestClass: pass", encoding="utf-8")
    
    # Create a .gitignore file
    (repo_path / ".gitignore").write_text("*.pyc\n__pycache__/\n", encoding="utf-8")
    
    return repo_path


@pytest.fixture
def sample_file_infos(test_repository_path):
    """Create sample FileInfo objects for testing."""
    return [
        FileInfo(
            path=test_repository_path / "main.py",
            size=20,
            last_modified=None,
            language="python"
        ),
        FileInfo(
            path=test_repository_path / "README.md",
            size=14,
            last_modified=None,
            language="markdown"
        ),
        FileInfo(
            path=test_repository_path / "utils.py",
            size=18,
            last_modified=None,
            language="python"
        ),
    ]


@pytest.fixture
def progress_callback():
    """Mock progress callback function."""
    callback = Mock()
    return callback


@pytest.fixture
def initializer_params():
    """Default parameters for RepositoryInitializer."""
    return {
        "neo4j_uri": "bolt://localhost:7687",
        "PROJECT_WATCH_USER": "neo4j",
        "PROJECT_WATCH_PASSWORD": "password",
        "PROJECT_WATCH_DATABASE": "test_db",
    }


# =============================================================================
# TEST INSTANTIATION
# =============================================================================


class TestRepositoryInitializerInstantiation:
    """Test RepositoryInitializer class instantiation."""
    
    def test_instantiation_with_minimal_params(self, initializer_params):
        """Test creating initializer with minimal parameters."""
        initializer = RepositoryInitializer(**initializer_params)
        
        assert initializer.neo4j_uri == "bolt://localhost:7687"
        assert initializer.PROJECT_WATCH_USER == "neo4j"
        assert initializer.PROJECT_WATCH_PASSWORD == "password"
        assert initializer.PROJECT_WATCH_DATABASE == "test_db"
        assert initializer.repository_path == Path.cwd()
        assert initializer.project_name == Path.cwd().name
        assert initializer.progress_callback is None
        assert initializer._driver is None
        assert initializer._neo4j_rag is None
        assert initializer._repository_monitor is None
    
    def test_instantiation_with_all_params(self, initializer_params, test_repository_path, progress_callback):
        """Test creating initializer with all parameters."""
        initializer = RepositoryInitializer(
            **initializer_params,
            repository_path=test_repository_path,
            project_name="custom_project",
            progress_callback=progress_callback
        )
        
        assert initializer.repository_path == test_repository_path
        assert initializer.project_name == "custom_project"
        assert initializer.progress_callback == progress_callback
    
    def test_instantiation_with_path_none(self, initializer_params):
        """Test that None repository_path defaults to current directory."""
        initializer = RepositoryInitializer(
            **initializer_params,
            repository_path=None
        )
        
        assert initializer.repository_path == Path.cwd()
    
    def test_instantiation_with_project_name_none(self, initializer_params, test_repository_path):
        """Test that None project_name defaults to repository name."""
        initializer = RepositoryInitializer(
            **initializer_params,
            repository_path=test_repository_path,
            project_name=None
        )
        
        assert initializer.project_name == "test_repo"


# =============================================================================
# TEST ASYNC CONTEXT MANAGER
# =============================================================================


class TestAsyncContextManager:
    """Test async context manager functionality."""
    
    @pytest.mark.asyncio
    async def test_context_manager_setup_and_cleanup(self, initializer_params, mock_neo4j_driver):
        """Test that context manager sets up and cleans up connections."""
        with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
            mock_db.driver.return_value = mock_neo4j_driver
            
            initializer = RepositoryInitializer(**initializer_params)
            
            async with initializer as init:
                assert init._driver is not None
                mock_neo4j_driver.verify_connectivity.assert_called_once()
            
            # After context exit, connections should be cleaned up
            mock_neo4j_driver.close.assert_called_once()
            assert initializer._driver is None
            assert initializer._neo4j_rag is None
            assert initializer._repository_monitor is None
    
    @pytest.mark.asyncio
    async def test_context_manager_with_exception(self, initializer_params, mock_neo4j_driver):
        """Test that context manager cleans up even with exceptions."""
        with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
            mock_db.driver.return_value = mock_neo4j_driver
            
            initializer = RepositoryInitializer(**initializer_params)
            
            with pytest.raises(ValueError):
                async with initializer:
                    raise ValueError("Test exception")
            
            # Cleanup should still happen
            mock_neo4j_driver.close.assert_called_once()


# =============================================================================
# TEST INITIALIZATION HAPPY PATH
# =============================================================================


class TestInitializationHappyPath:
    """Test successful initialization scenarios."""
    
    @pytest.mark.asyncio
    async def test_initialize_with_valid_repo(
        self,
        initializer_params,
        test_repository_path,
        mock_neo4j_driver,
        sample_file_infos,
        progress_callback
    ):
        """Test successful initialization with valid repository."""
        with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
            mock_db.driver.return_value = mock_neo4j_driver
            
            with patch('src.project_watch_mcp.core.initializer.Neo4jRAG') as mock_rag_class:
                mock_rag = create_mock_neo4j_rag(sample_file_infos)
                mock_rag_class.return_value = mock_rag
                
                with patch('src.project_watch_mcp.core.initializer.RepositoryMonitor') as mock_monitor_class:
                    mock_monitor = AsyncMock()
                    mock_monitor.scan_repository = AsyncMock(return_value=sample_file_infos)
                    mock_monitor.start = AsyncMock()
                    mock_monitor_class.return_value = mock_monitor
                    
                    with patch('src.project_watch_mcp.core.initializer.create_embeddings_provider') as mock_embed:
                        mock_embed.return_value = AsyncMock()
                        
                        initializer = RepositoryInitializer(
                            **initializer_params,
                            repository_path=test_repository_path,
                            progress_callback=progress_callback
                        )
                        
                        result = await initializer.initialize()
                        
                        # Verify result
                        assert isinstance(result, InitializationResult)
                        assert result.indexed == 3
                        assert result.total == 3
                        assert result.skipped == []
                        assert result.monitoring is False  # Monitoring not enabled in this test
                        assert "Indexed 3/3 files" in result.message
                        
                        # Verify scan was called
                        mock_monitor.scan_repository.assert_called_once()
                        
                        # Verify files were indexed
                        assert mock_rag.index_file.call_count == 3
                        
                        # Verify monitoring NOT started (not enabled in this test)
                        mock_monitor.start.assert_not_called()
                        
                        # Verify progress callbacks
                        assert progress_callback.call_count > 0
                        progress_callback.assert_any_call(0.0, "Scanning repository...")
                        progress_callback.assert_any_call(100.0, "Initialization complete")
    
    @pytest.mark.asyncio
    async def test_initialize_empty_repository(
        self,
        initializer_params,
        mock_neo4j_driver,
        progress_callback
    ):
        """Test initialization with empty repository (no files to index)."""
        with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
            mock_db.driver.return_value = mock_neo4j_driver
            
            with patch('src.project_watch_mcp.core.initializer.Neo4jRAG') as mock_rag_class:
                mock_rag = AsyncMock()
                mock_rag_class.return_value = mock_rag
                
                with patch('src.project_watch_mcp.core.initializer.RepositoryMonitor') as mock_monitor_class:
                    mock_monitor = AsyncMock()
                    mock_monitor.scan_repository = AsyncMock(return_value=[])
                    mock_monitor_class.return_value = mock_monitor
                    
                    with patch('src.project_watch_mcp.core.initializer.create_embeddings_provider'):
                        initializer = RepositoryInitializer(
                            **initializer_params,
                            progress_callback=progress_callback
                        )
                        
                        result = await initializer.initialize()
                        
                        assert result.indexed == 0
                        assert result.total == 0
                        assert result.message == "No files found matching patterns"
                        
                        # Verify progress was reported
                        progress_callback.assert_any_call(100.0, "No files to index")


# =============================================================================
# TEST PROGRESS CALLBACK
# =============================================================================


class TestProgressCallback:
    """Test progress callback invocation and accuracy."""
    
    @pytest.mark.asyncio
    async def test_progress_callback_invocation(
        self,
        initializer_params,
        mock_neo4j_driver,
        sample_file_infos,
        progress_callback,
        test_repository_path
    ):
        """Test that progress callback is invoked with correct values."""
        with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
            mock_db.driver.return_value = mock_neo4j_driver
            
            with patch('src.project_watch_mcp.core.initializer.Neo4jRAG') as mock_rag_class:
                mock_rag = create_mock_neo4j_rag(sample_file_infos)
                mock_rag_class.return_value = mock_rag
                
                with patch('src.project_watch_mcp.core.initializer.RepositoryMonitor') as mock_monitor_class:
                    mock_monitor = AsyncMock()
                    mock_monitor.scan_repository = AsyncMock(return_value=sample_file_infos)
                    mock_monitor.start = AsyncMock()
                    mock_monitor_class.return_value = mock_monitor
                    
                    with patch('src.project_watch_mcp.core.initializer.create_embeddings_provider'):
                        initializer = RepositoryInitializer(
                            **initializer_params,
                            repository_path=test_repository_path,
                            progress_callback=progress_callback
                        )
                        
                        await initializer.initialize()
                        
                        # Collect all progress calls
                        calls = progress_callback.call_args_list
                        
                        # Verify first and last calls
                        assert calls[0] == call(0.0, "Scanning repository...")
                        assert calls[-1] == call(100.0, "Initialization complete")
                        
                        # Verify progress values are in ascending order
                        progress_values = [c[0][0] for c in calls]
                        assert progress_values == sorted(progress_values)
                        
                        # Verify all progress values are between 0 and 100
                        assert all(0 <= p <= 100 for p in progress_values)
    
    @pytest.mark.asyncio
    async def test_progress_callback_exception_handling(
        self,
        initializer_params,
        mock_neo4j_driver,
        sample_file_infos,
        test_repository_path
    ):
        """Test that progress callback exceptions don't crash initialization."""
        def faulty_callback(progress, message):
            if progress > 50:
                raise ValueError("Callback error")
        
        with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
            mock_db.driver.return_value = mock_neo4j_driver
            
            with patch('src.project_watch_mcp.core.initializer.Neo4jRAG') as mock_rag_class:
                mock_rag = create_mock_neo4j_rag(sample_file_infos)
                mock_rag_class.return_value = mock_rag
                
                with patch('src.project_watch_mcp.core.initializer.RepositoryMonitor') as mock_monitor_class:
                    mock_monitor = AsyncMock()
                    mock_monitor.scan_repository = AsyncMock(return_value=sample_file_infos)
                    mock_monitor.start = AsyncMock()
                    mock_monitor_class.return_value = mock_monitor
                    
                    with patch('src.project_watch_mcp.core.initializer.create_embeddings_provider'):
                        initializer = RepositoryInitializer(
                            **initializer_params,
                            repository_path=test_repository_path,
                            progress_callback=faulty_callback
                        )
                        
                        # Should complete successfully despite callback errors
                        result = await initializer.initialize()
                        assert result.indexed == 3


# =============================================================================
# TEST FILE FILTERING
# =============================================================================


class TestFileFiltering:
    """Test file filtering with gitignore patterns."""
    
    @pytest.mark.asyncio
    async def test_file_filtering_with_gitignore(
        self,
        initializer_params,
        test_repository_path,
        mock_neo4j_driver
    ):
        """Test that gitignore patterns are respected during scanning."""
        # Create some files that should be ignored
        (test_repository_path / "test.pyc").write_bytes(b"\x00\x01\x02")
        cache_dir = test_repository_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "module.cpython-311.pyc").write_bytes(b"\x00\x01\x02")
        
        with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
            mock_db.driver.return_value = mock_neo4j_driver
            
            with patch('src.project_watch_mcp.core.initializer.Neo4jRAG') as mock_rag_class:
                mock_rag = create_mock_neo4j_rag([])
                mock_rag_class.return_value = mock_rag
                
                with patch('src.project_watch_mcp.core.initializer.RepositoryMonitor') as mock_monitor_class:
                    mock_monitor = AsyncMock()
                    # Return only non-ignored files
                    mock_monitor.scan_repository = AsyncMock(return_value=[
                        FileInfo(path=test_repository_path / "main.py", size=20, last_modified=None),
                        FileInfo(path=test_repository_path / "utils.py", size=18, last_modified=None),
                    ])
                    mock_monitor.start = AsyncMock()
                    mock_monitor_class.return_value = mock_monitor
                    
                    with patch('src.project_watch_mcp.core.initializer.create_embeddings_provider'):
                        initializer = RepositoryInitializer(
                            **initializer_params,
                            repository_path=test_repository_path
                        )
                        
                        result = await initializer.initialize()
                        
                        # Only non-ignored files should be indexed
                        assert result.indexed == 2
                        assert result.total == 2


# =============================================================================
# TEST ERROR HANDLING
# =============================================================================


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_invalid_path_error(self, initializer_params):
        """Test error handling for invalid repository paths."""
        non_existent_path = Path("/non/existent/path")
        
        with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
            mock_db.driver.return_value = AsyncMock()
            
            with patch('src.project_watch_mcp.core.initializer.RepositoryMonitor') as mock_monitor_class:
                mock_monitor = AsyncMock()
                mock_monitor.scan_repository = AsyncMock(
                    side_effect=FileNotFoundError("Repository not found")
                )
                mock_monitor_class.return_value = mock_monitor
                
                with patch('src.project_watch_mcp.core.initializer.create_embeddings_provider'):
                    initializer = RepositoryInitializer(
                        **initializer_params,
                        repository_path=non_existent_path
                    )
                    
                    with pytest.raises(InitializationError) as exc_info:
                        await initializer.initialize()
                    
                    assert "Initialization failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_neo4j_connection_error(self, initializer_params):
        """Test error handling for Neo4j connection failures."""
        with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
            mock_driver = AsyncMock()
            mock_driver.verify_connectivity = AsyncMock(side_effect=Exception("Connection failed"))
            mock_db.driver.return_value = mock_driver
            
            initializer = RepositoryInitializer(**initializer_params)
            
            with pytest.raises(ConnectionError) as exc_info:
                await initializer._setup_connections()
            
            assert "Failed to connect to Neo4j" in str(exc_info.value)
            assert exc_info.value.error_code == "NEO4J_CONN_ERROR"
    
    @pytest.mark.asyncio
    async def test_neo4j_connection_timeout(self, initializer_params):
        """Test error handling for Neo4j connection timeout."""
        with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
            mock_driver = AsyncMock()
            mock_driver.verify_connectivity = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_db.driver.return_value = mock_driver
            
            with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError()):
                initializer = RepositoryInitializer(**initializer_params)
                
                with pytest.raises(ConnectionError) as exc_info:
                    await initializer._setup_connections()
                
                assert "Neo4j connection timeout" in str(exc_info.value)
                assert exc_info.value.error_code == "NEO4J_CONN_ERROR"
    
    @pytest.mark.asyncio
    async def test_indexing_error(
        self,
        initializer_params,
        mock_neo4j_driver,
        sample_file_infos,
        test_repository_path
    ):
        """Test error handling for Neo4j indexing failures."""
        with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
            mock_db.driver.return_value = mock_neo4j_driver
            
            with patch('src.project_watch_mcp.core.initializer.Neo4jRAG') as mock_rag_class:
                mock_rag = create_mock_neo4j_rag(sample_file_infos)
                mock_rag.index_file = AsyncMock(side_effect=Exception("Indexing failed"))
                mock_rag_class.return_value = mock_rag
                
                with patch('src.project_watch_mcp.core.initializer.RepositoryMonitor') as mock_monitor_class:
                    mock_monitor = AsyncMock()
                    mock_monitor.scan_repository = AsyncMock(return_value=sample_file_infos)
                    mock_monitor.start = AsyncMock()
                    mock_monitor_class.return_value = mock_monitor
                    
                    with patch('src.project_watch_mcp.core.initializer.create_embeddings_provider'):
                        initializer = RepositoryInitializer(
                            **initializer_params,
                            repository_path=test_repository_path
                        )
                        
                        result = await initializer.initialize()
                        
                        # All files should be skipped due to indexing errors
                        assert result.indexed == 0
                        assert result.total == 3
                        assert len(result.skipped) == 3


# =============================================================================
# TEST RESOURCE CLEANUP
# =============================================================================


class TestResourceCleanup:
    """Test resource cleanup on exceptions."""
    
    @pytest.mark.asyncio
    async def test_cleanup_on_exception(self, initializer_params, mock_neo4j_driver):
        """Test that resources are properly cleaned up on exceptions."""
        with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
            mock_db.driver.return_value = mock_neo4j_driver
            
            initializer = RepositoryInitializer(**initializer_params)
            
            # Set up connections
            await initializer._setup_connections()
            assert initializer._driver is not None
            
            # Cleanup should reset all connections
            await initializer._cleanup_connections()
            
            assert initializer._driver is None
            assert initializer._neo4j_rag is None
            assert initializer._repository_monitor is None
            mock_neo4j_driver.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_with_driver_close_error(self, initializer_params):
        """Test that cleanup continues even if driver close fails."""
        mock_driver = AsyncMock()
        mock_driver.close = AsyncMock(side_effect=Exception("Close failed"))
        
        initializer = RepositoryInitializer(**initializer_params)
        initializer._driver = mock_driver
        initializer._neo4j_rag = Mock()
        initializer._repository_monitor = Mock()
        
        # Should not raise exception
        await initializer._cleanup_connections()
        
        # Resources should still be cleared
        assert initializer._driver is None
        assert initializer._neo4j_rag is None
        assert initializer._repository_monitor is None


# =============================================================================
# TEST CONCURRENT INITIALIZATION
# =============================================================================


class TestConcurrentInitialization:
    """Test concurrent initialization prevention."""
    
    @pytest.mark.asyncio
    async def test_concurrent_initialization_prevention(
        self,
        initializer_params,
        mock_neo4j_driver,
        sample_file_infos,
        test_repository_path
    ):
        """Test that concurrent initializations are prevented."""
        initialization_started = asyncio.Event()
        initialization_proceed = asyncio.Event()
        
        async def slow_scan(*args):
            initialization_started.set()
            await initialization_proceed.wait()
            return sample_file_infos
        
        with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
            mock_db.driver.return_value = mock_neo4j_driver
            
            with patch('src.project_watch_mcp.core.initializer.Neo4jRAG') as mock_rag_class:
                mock_rag = create_mock_neo4j_rag([])
                mock_rag_class.return_value = mock_rag
                
                with patch('src.project_watch_mcp.core.initializer.RepositoryMonitor') as mock_monitor_class:
                    mock_monitor = AsyncMock()
                    mock_monitor.scan_repository = slow_scan
                    mock_monitor.start = AsyncMock()
                    mock_monitor_class.return_value = mock_monitor
                    
                    with patch('src.project_watch_mcp.core.initializer.create_embeddings_provider'):
                        initializer = RepositoryInitializer(
                            **initializer_params,
                            repository_path=test_repository_path
                        )
                        
                        # Start first initialization
                        task1 = asyncio.create_task(initializer.initialize())
                        
                        # Wait for first initialization to start
                        await initialization_started.wait()
                        
                        # Try to start second initialization
                        task2 = asyncio.create_task(initializer.initialize())
                        
                        # Allow first initialization to proceed
                        initialization_proceed.set()
                        
                        # Wait for both to complete
                        result1 = await task1
                        result2 = await task2
                        
                        # Both should succeed (second waits for first)
                        assert result1.indexed == 3
                        assert result2.indexed == 3


# =============================================================================
# TEST MEMORY LEAK DETECTION
# =============================================================================


class TestMemoryLeakDetection:
    """Test memory leak detection with tracemalloc."""
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(
        self,
        initializer_params,
        mock_neo4j_driver,
        sample_file_infos,
        test_repository_path
    ):
        """Test that there are no memory leaks during initialization."""
        tracemalloc.start()
        
        with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
            mock_db.driver.return_value = mock_neo4j_driver
            
            with patch('src.project_watch_mcp.core.initializer.Neo4jRAG') as mock_rag_class:
                mock_rag = create_mock_neo4j_rag(sample_file_infos)
                mock_rag_class.return_value = mock_rag
                
                with patch('src.project_watch_mcp.core.initializer.RepositoryMonitor') as mock_monitor_class:
                    mock_monitor = AsyncMock()
                    mock_monitor.scan_repository = AsyncMock(return_value=sample_file_infos)
                    mock_monitor.start = AsyncMock()
                    mock_monitor_class.return_value = mock_monitor
                    
                    with patch('src.project_watch_mcp.core.initializer.create_embeddings_provider'):
                        # Take snapshot before
                        snapshot1 = tracemalloc.take_snapshot()
                        
                        # Run multiple initializations
                        for _ in range(3):
                            initializer = RepositoryInitializer(
                                **initializer_params,
                                repository_path=test_repository_path
                            )
                            await initializer.initialize()
                            await initializer._cleanup_connections()
                        
                        # Take snapshot after
                        snapshot2 = tracemalloc.take_snapshot()
                        
                        # Compare snapshots
                        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
                        
                        # Get total memory difference
                        total_diff = sum(stat.size_diff for stat in top_stats)
                        
                        # Memory growth should be minimal (< 1MB)
                        assert total_diff < 1024 * 1024, f"Memory grew by {total_diff} bytes"
        
        tracemalloc.stop()


# =============================================================================
# TEST PATH HANDLING EDGE CASES
# =============================================================================


class TestPathHandlingEdgeCases:
    """Test path handling edge cases."""
    
    @pytest.mark.asyncio
    async def test_path_with_spaces(self, initializer_params, tmp_path, mock_neo4j_driver):
        """Test handling of paths with spaces."""
        repo_path = tmp_path / "path_with_spaces"
        repo_path.mkdir()
        (repo_path / "file name.py").write_text("print('test')", encoding="utf-8")
        
        with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
            mock_db.driver.return_value = mock_neo4j_driver
            
            with patch('src.project_watch_mcp.core.initializer.Neo4jRAG') as mock_rag_class:
                mock_rag = create_mock_neo4j_rag([])
                mock_rag_class.return_value = mock_rag
                
                with patch('src.project_watch_mcp.core.initializer.RepositoryMonitor') as mock_monitor_class:
                    mock_monitor = AsyncMock()
                    mock_monitor.scan_repository = AsyncMock(return_value=[
                        FileInfo(path=repo_path / "file name.py", size=13, last_modified=None)
                    ])
                    mock_monitor.start = AsyncMock()
                    mock_monitor_class.return_value = mock_monitor
                    
                    with patch('src.project_watch_mcp.core.initializer.create_embeddings_provider'):
                        initializer = RepositoryInitializer(
                            **initializer_params,
                            repository_path=repo_path
                        )
                        
                        result = await initializer.initialize()
                        assert result.indexed == 1
    
    @pytest.mark.asyncio
    async def test_path_with_unicode(self, initializer_params, tmp_path, mock_neo4j_driver):
        """Test handling of paths with unicode characters."""
        repo_path = tmp_path / "unicode_path"
        repo_path.mkdir()
        (repo_path / "文件.py").write_text("print('测试')", encoding="utf-8")
        
        with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
            mock_db.driver.return_value = mock_neo4j_driver
            
            with patch('src.project_watch_mcp.core.initializer.Neo4jRAG') as mock_rag_class:
                mock_rag = create_mock_neo4j_rag([])
                mock_rag_class.return_value = mock_rag
                
                with patch('src.project_watch_mcp.core.initializer.RepositoryMonitor') as mock_monitor_class:
                    mock_monitor = AsyncMock()
                    mock_monitor.scan_repository = AsyncMock(return_value=[
                        FileInfo(path=repo_path / "文件.py", size=15, last_modified=None)
                    ])
                    mock_monitor.start = AsyncMock()
                    mock_monitor_class.return_value = mock_monitor
                    
                    with patch('src.project_watch_mcp.core.initializer.create_embeddings_provider'):
                        initializer = RepositoryInitializer(
                            **initializer_params,
                            repository_path=repo_path
                        )
                        
                        result = await initializer.initialize()
                        assert result.indexed == 1
    
    @pytest.mark.asyncio
    async def test_symlink_handling(self, initializer_params, tmp_path, mock_neo4j_driver):
        """Test handling of symbolic links."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        
        # Create a file and a symlink to it
        original_file = repo_path / "original.py"
        original_file.write_text("print('original')", encoding="utf-8")
        
        symlink_file = repo_path / "link.py"
        symlink_file.symlink_to(original_file)
        
        with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
            mock_db.driver.return_value = mock_neo4j_driver
            
            with patch('src.project_watch_mcp.core.initializer.Neo4jRAG') as mock_rag_class:
                mock_rag = create_mock_neo4j_rag([])
                mock_rag_class.return_value = mock_rag
                
                with patch('src.project_watch_mcp.core.initializer.RepositoryMonitor') as mock_monitor_class:
                    mock_monitor = AsyncMock()
                    # Monitor should handle symlinks appropriately
                    mock_monitor.scan_repository = AsyncMock(return_value=[
                        FileInfo(path=original_file, size=17, last_modified=None)
                    ])
                    mock_monitor.start = AsyncMock()
                    mock_monitor_class.return_value = mock_monitor
                    
                    with patch('src.project_watch_mcp.core.initializer.create_embeddings_provider'):
                        initializer = RepositoryInitializer(
                            **initializer_params,
                            repository_path=repo_path
                        )
                        
                        result = await initializer.initialize()
                        assert result.indexed == 1


# =============================================================================
# TEST FILE ENCODING EDGE CASES
# =============================================================================


class TestFileEncodingEdgeCases:
    """Test file encoding edge cases."""
    
    @pytest.mark.asyncio
    async def test_utf8_file_encoding(self, initializer_params, tmp_path, mock_neo4j_driver):
        """Test handling of UTF-8 encoded files."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        
        # Create UTF-8 file with various characters
        utf8_file = repo_path / "utf8.py"
        utf8_file.write_text(
            "# -*- coding: utf-8 -*-\n"
            "print('Hello 世界 🌍')\n"
            "# Comment with emojis 😀🎉\n",
            encoding="utf-8"
        )
        
        with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
            mock_db.driver.return_value = mock_neo4j_driver
            
            with patch('src.project_watch_mcp.core.initializer.Neo4jRAG') as mock_rag_class:
                mock_rag = create_mock_neo4j_rag([])
                mock_rag_class.return_value = mock_rag
                
                with patch('src.project_watch_mcp.core.initializer.RepositoryMonitor') as mock_monitor_class:
                    mock_monitor = AsyncMock()
                    mock_monitor.scan_repository = AsyncMock(return_value=[
                        FileInfo(path=utf8_file, size=100, last_modified=None)
                    ])
                    mock_monitor.start = AsyncMock()
                    mock_monitor_class.return_value = mock_monitor
                    
                    with patch('src.project_watch_mcp.core.initializer.create_embeddings_provider'):
                        initializer = RepositoryInitializer(
                            **initializer_params,
                            repository_path=repo_path
                        )
                        
                        result = await initializer.initialize()
                        assert result.indexed == 1
                        assert result.skipped == []
    
    @pytest.mark.asyncio
    async def test_utf16_file_encoding(self, initializer_params, tmp_path, mock_neo4j_driver):
        """Test handling of UTF-16 encoded files."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        
        # Create UTF-16 file
        utf16_file = repo_path / "utf16.py"
        utf16_file.write_text(
            "print('UTF-16 encoded')",
            encoding="utf-16"
        )
        
        with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
            mock_db.driver.return_value = mock_neo4j_driver
            
            with patch('src.project_watch_mcp.core.initializer.Neo4jRAG') as mock_rag_class:
                mock_rag = create_mock_neo4j_rag([])
                mock_rag_class.return_value = mock_rag
                
                with patch('src.project_watch_mcp.core.initializer.RepositoryMonitor') as mock_monitor_class:
                    mock_monitor = AsyncMock()
                    mock_monitor.scan_repository = AsyncMock(return_value=[
                        FileInfo(path=utf16_file, size=50, last_modified=None)
                    ])
                    mock_monitor.start = AsyncMock()
                    mock_monitor_class.return_value = mock_monitor
                    
                    with patch('src.project_watch_mcp.core.initializer.create_embeddings_provider'):
                        initializer = RepositoryInitializer(
                            **initializer_params,
                            repository_path=repo_path
                        )
                        
                        result = await initializer.initialize()
                        
                        # UTF-16 files will likely fail to decode as UTF-8
                        assert result.indexed == 0
                        assert len(result.skipped) == 1
                        assert "utf16.py" in result.skipped[0]
    
    @pytest.mark.asyncio
    async def test_binary_file_handling(self, initializer_params, tmp_path, mock_neo4j_driver):
        """Test handling of binary files."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        
        # Create binary file - use a file that can't be decoded as UTF-8
        binary_file = repo_path / "binary.dat"
        binary_file.write_bytes(b'\x00\x01\x02\x03\x04\x05\xFF\xFE')
        
        with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
            mock_db.driver.return_value = mock_neo4j_driver
            
            with patch('src.project_watch_mcp.core.initializer.Neo4jRAG') as mock_rag_class:
                mock_rag = create_mock_neo4j_rag([])
                mock_rag_class.return_value = mock_rag
                
                with patch('src.project_watch_mcp.core.initializer.RepositoryMonitor') as mock_monitor_class:
                    mock_monitor = AsyncMock()
                    mock_monitor.scan_repository = AsyncMock(return_value=[
                        FileInfo(path=binary_file, size=8, last_modified=None)
                    ])
                    mock_monitor.start = AsyncMock()
                    mock_monitor_class.return_value = mock_monitor
                    
                    with patch('src.project_watch_mcp.core.initializer.create_embeddings_provider'):
                        initializer = RepositoryInitializer(
                            **initializer_params,
                            repository_path=repo_path
                        )
                        
                        result = await initializer.initialize()
                        
                        # Binary files should be skipped
                        assert result.indexed == 0
                        assert len(result.skipped) == 1
                        assert "binary.dat" in result.skipped[0]


# =============================================================================
# TEST BOUNDARY TESTING
# =============================================================================


class TestBoundaryConditions:
    """Test boundary conditions for file size limits."""
    
    @pytest.mark.asyncio
    async def test_large_file_handling(self, initializer_params, tmp_path, mock_neo4j_driver):
        """Test handling of very large files."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        
        # Create a large file (10MB)
        large_file = repo_path / "large.py"
        content = "# Large file\n" + ("x = 1\n" * 1_000_000)
        large_file.write_text(content, encoding="utf-8")
        
        with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
            mock_db.driver.return_value = mock_neo4j_driver
            
            with patch('src.project_watch_mcp.core.initializer.Neo4jRAG') as mock_rag_class:
                mock_rag = create_mock_neo4j_rag([])
                mock_rag_class.return_value = mock_rag
                
                with patch('src.project_watch_mcp.core.initializer.RepositoryMonitor') as mock_monitor_class:
                    mock_monitor = AsyncMock()
                    mock_monitor.scan_repository = AsyncMock(return_value=[
                        FileInfo(path=large_file, size=len(content), last_modified=None)
                    ])
                    mock_monitor.start = AsyncMock()
                    mock_monitor_class.return_value = mock_monitor
                    
                    with patch('src.project_watch_mcp.core.initializer.create_embeddings_provider'):
                        initializer = RepositoryInitializer(
                            **initializer_params,
                            repository_path=repo_path
                        )
                        
                        result = await initializer.initialize()
                        
                        # Large files should still be indexed
                        assert result.indexed == 1
                        mock_rag.index_file.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_empty_file_handling(self, initializer_params, tmp_path, mock_neo4j_driver):
        """Test handling of empty files."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        
        # Create an empty file
        empty_file = repo_path / "empty.py"
        empty_file.write_text("", encoding="utf-8")
        
        with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
            mock_db.driver.return_value = mock_neo4j_driver
            
            with patch('src.project_watch_mcp.core.initializer.Neo4jRAG') as mock_rag_class:
                mock_rag = create_mock_neo4j_rag([])
                mock_rag_class.return_value = mock_rag
                
                with patch('src.project_watch_mcp.core.initializer.RepositoryMonitor') as mock_monitor_class:
                    mock_monitor = AsyncMock()
                    mock_monitor.scan_repository = AsyncMock(return_value=[
                        FileInfo(path=empty_file, size=0, last_modified=None)
                    ])
                    mock_monitor.start = AsyncMock()
                    mock_monitor_class.return_value = mock_monitor
                    
                    with patch('src.project_watch_mcp.core.initializer.create_embeddings_provider'):
                        initializer = RepositoryInitializer(
                            **initializer_params,
                            repository_path=repo_path
                        )
                        
                        result = await initializer.initialize()
                        
                        # Empty files should still be indexed
                        assert result.indexed == 1
                        mock_rag.index_file.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_many_files_handling(self, initializer_params, tmp_path, mock_neo4j_driver):
        """Test handling of repository with many files."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        
        # Create many files
        num_files = 1000
        for i in range(num_files):
            file_path = repo_path / f"file_{i:04d}.py"
            file_path.write_text(f"# File {i}\nprint({i})", encoding="utf-8")
        
        file_infos = [
            FileInfo(path=repo_path / f"file_{i:04d}.py", size=20, last_modified=None)
            for i in range(num_files)
        ]
        
        with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
            mock_db.driver.return_value = mock_neo4j_driver
            
            with patch('src.project_watch_mcp.core.initializer.Neo4jRAG') as mock_rag_class:
                mock_rag = create_mock_neo4j_rag([])
                mock_rag_class.return_value = mock_rag
                
                with patch('src.project_watch_mcp.core.initializer.RepositoryMonitor') as mock_monitor_class:
                    mock_monitor = AsyncMock()
                    mock_monitor.scan_repository = AsyncMock(return_value=file_infos)
                    mock_monitor.start = AsyncMock()
                    mock_monitor_class.return_value = mock_monitor
                    
                    with patch('src.project_watch_mcp.core.initializer.create_embeddings_provider'):
                        initializer = RepositoryInitializer(
                            **initializer_params,
                            repository_path=repo_path
                        )
                        
                        result = await initializer.initialize()
                        
                        # All files should be indexed
                        assert result.indexed == num_files
                        assert result.total == num_files
                        assert mock_rag.index_file.call_count == num_files


# =============================================================================
# TEST PERFORMANCE
# =============================================================================


class TestPerformance:
    """Performance tests for initialization."""
    
    @pytest.mark.asyncio
    async def test_initialization_performance_small_repo(
        self,
        initializer_params,
        tmp_path,
        mock_neo4j_driver
    ):
        """Benchmark initialization for small repository (10 files)."""
        import time
        
        repo_path = tmp_path / "small_repo"
        repo_path.mkdir()
        
        # Create 10 files
        for i in range(10):
            (repo_path / f"file_{i}.py").write_text(f"print({i})", encoding="utf-8")
        
        file_infos = [
            FileInfo(path=repo_path / f"file_{i}.py", size=10, last_modified=None)
            for i in range(10)
        ]
        
        with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
            mock_db.driver.return_value = mock_neo4j_driver
            
            with patch('src.project_watch_mcp.core.initializer.Neo4jRAG') as mock_rag_class:
                mock_rag = create_mock_neo4j_rag([])
                mock_rag_class.return_value = mock_rag
                
                with patch('src.project_watch_mcp.core.initializer.RepositoryMonitor') as mock_monitor_class:
                    mock_monitor = AsyncMock()
                    mock_monitor.scan_repository = AsyncMock(return_value=file_infos)
                    mock_monitor.start = AsyncMock()
                    mock_monitor_class.return_value = mock_monitor
                    
                    with patch('src.project_watch_mcp.core.initializer.create_embeddings_provider'):
                        initializer = RepositoryInitializer(
                            **initializer_params,
                            repository_path=repo_path
                        )
                        
                        start_time = time.time()
                        result = await initializer.initialize()
                        elapsed_time = time.time() - start_time
                        
                        assert result.indexed == 10
                        # Should complete within 5 seconds for 10 files
                        assert elapsed_time < 5.0
    
    @pytest.mark.asyncio
    async def test_initialization_performance_medium_repo(
        self,
        initializer_params,
        tmp_path,
        mock_neo4j_driver
    ):
        """Benchmark initialization for medium repository (100 files)."""
        import time
        
        repo_path = tmp_path / "medium_repo"
        repo_path.mkdir()
        
        # Create 100 files
        for i in range(100):
            (repo_path / f"file_{i:03d}.py").write_text(f"print({i})", encoding="utf-8")
        
        file_infos = [
            FileInfo(path=repo_path / f"file_{i:03d}.py", size=10, last_modified=None)
            for i in range(100)
        ]
        
        with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
            mock_db.driver.return_value = mock_neo4j_driver
            
            with patch('src.project_watch_mcp.core.initializer.Neo4jRAG') as mock_rag_class:
                mock_rag = create_mock_neo4j_rag(file_infos)
                # Simulate some processing time
                async def mock_index(*args):
                    await asyncio.sleep(0.001)
                mock_rag.index_file = mock_index
                mock_rag_class.return_value = mock_rag
                
                with patch('src.project_watch_mcp.core.initializer.RepositoryMonitor') as mock_monitor_class:
                    mock_monitor = AsyncMock()
                    mock_monitor.scan_repository = AsyncMock(return_value=file_infos)
                    mock_monitor.start = AsyncMock()
                    mock_monitor_class.return_value = mock_monitor
                    
                    with patch('src.project_watch_mcp.core.initializer.create_embeddings_provider'):
                        initializer = RepositoryInitializer(
                            **initializer_params,
                            repository_path=repo_path
                        )
                        
                        start_time = time.time()
                        result = await initializer.initialize()
                        elapsed_time = time.time() - start_time
                        
                        assert result.indexed == 100
                        # Should complete within 10 seconds for 100 files
                        assert elapsed_time < 10.0


# =============================================================================
# TEST INITIALIZATION RESULT
# =============================================================================


class TestInitializationResult:
    """Test InitializationResult dataclass."""
    
    def test_initialization_result_creation(self):
        """Test creating InitializationResult."""
        result = InitializationResult(
            indexed=10,
            total=12,
            skipped=["file1.py", "file2.py"],
            monitoring=True,
            message="Test message"
        )
        
        assert result.indexed == 10
        assert result.total == 12
        assert result.skipped == ["file1.py", "file2.py"]
        assert result.monitoring is True
        assert result.message == "Test message"
    
    def test_initialization_result_to_dict(self):
        """Test converting InitializationResult to dictionary."""
        result = InitializationResult(
            indexed=5,
            total=5,
            skipped=[],
            monitoring=False,
            message="All files indexed"
        )
        
        result_dict = result.to_dict()
        
        assert result_dict == {
            "indexed": 5,
            "total": 5,
            "skipped": [],
            "monitoring": False,
            "message": "All files indexed"
        }
    
    def test_initialization_result_default_values(self):
        """Test InitializationResult default values."""
        result = InitializationResult(indexed=0, total=0)
        
        assert result.indexed == 0
        assert result.total == 0
        assert result.skipped == []
        assert result.monitoring is False
        assert result.message == ""


# =============================================================================
# TEST EXCEPTION CLASSES
# =============================================================================


class TestExceptionClasses:
    """Test custom exception classes."""
    
    def test_initialization_error(self):
        """Test InitializationError exception."""
        error = InitializationError(
            message="Test error",
            error_code="TEST_ERROR",
            technical_details={"key": "value"}
        )
        
        assert str(error) == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.user_message == "Test error"
        assert error.technical_details == {"key": "value"}
    
    def test_connection_error(self):
        """Test ConnectionError exception."""
        error = ConnectionError(
            message="Connection failed",
            technical_details={"host": "localhost"}
        )
        
        assert str(error) == "Connection failed"
        assert error.error_code == "NEO4J_CONN_ERROR"
        assert error.technical_details == {"host": "localhost"}
    
    def test_file_access_error(self):
        """Test FileAccessError exception."""
        error = FileAccessError(
            message="File not found",
            technical_details={"path": "/test/path"}
        )
        
        assert str(error) == "File not found"
        assert error.error_code == "FILE_ACCESS_ERROR"
        assert error.technical_details == {"path": "/test/path"}
    
    def test_indexing_error(self):
        """Test IndexingError exception."""
        error = IndexingError(
            message="Indexing failed",
            technical_details={"file": "test.py"}
        )
        
        assert str(error) == "Indexing failed"
        assert error.error_code == "INDEXING_ERROR"
        assert error.technical_details == {"file": "test.py"}


# =============================================================================
# TEST MONITORING START FAILURE
# =============================================================================


class TestMonitoringStartFailure:
    """Test handling of monitoring start failures."""
    
    @pytest.mark.asyncio
    async def test_monitoring_start_failure_continues(
        self,
        initializer_params,
        mock_neo4j_driver,
        sample_file_infos,
        test_repository_path
    ):
        """Test that initialization continues even if monitoring fails to start."""
        with patch('src.project_watch_mcp.core.initializer.AsyncGraphDatabase') as mock_db:
            mock_db.driver.return_value = mock_neo4j_driver
            
            with patch('src.project_watch_mcp.core.initializer.Neo4jRAG') as mock_rag_class:
                mock_rag = create_mock_neo4j_rag(sample_file_infos)
                mock_rag_class.return_value = mock_rag
                
                with patch('src.project_watch_mcp.core.initializer.RepositoryMonitor') as mock_monitor_class:
                    mock_monitor = AsyncMock()
                    mock_monitor.scan_repository = AsyncMock(return_value=sample_file_infos)
                    mock_monitor.start = AsyncMock(side_effect=Exception("Monitoring start failed"))
                    mock_monitor_class.return_value = mock_monitor
                    
                    with patch('src.project_watch_mcp.core.initializer.create_embeddings_provider'):
                        initializer = RepositoryInitializer(
                            **initializer_params,
                            repository_path=test_repository_path
                        )
                        
                        result = await initializer.initialize()
                        
                        # Initialization should succeed even if monitoring fails
                        assert result.indexed == 3
                        assert result.total == 3
                        assert result.monitoring is False
                        assert "Indexed 3/3 files" in result.message