"""
Integration tests for incremental indexing functionality.

This module provides comprehensive integration tests for the incremental indexing
feature, verifying performance improvements and correctness across various scenarios.
"""

import asyncio
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase, AsyncSession

from src.project_watch_mcp.neo4j_rag import Neo4jRAG, CodeFile
from src.project_watch_mcp.repository_monitor import FileInfo, RepositoryMonitor
from src.project_watch_mcp.core.initializer import RepositoryInitializer
from src.project_watch_mcp.config import ProjectWatchConfig


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
async def temp_repository():
    """Create a temporary repository with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        
        # Create test files
        (repo_path / "file1.py").write_text("def test1(): pass")
        (repo_path / "file2.py").write_text("def test2(): pass")
        (repo_path / "file3.js").write_text("function test3() {}")
        (repo_path / "subdir").mkdir()
        (repo_path / "subdir" / "file4.py").write_text("def test4(): pass")
        
        yield repo_path


@pytest.fixture
async def neo4j_config():
    """Provide Neo4j configuration for tests."""
    return {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "password"),
        "database": os.getenv("NEO4J_DATABASE", "neo4j")
    }


@pytest.fixture
async def clean_neo4j(neo4j_config):
    """Ensure Neo4j is clean before and after tests."""
    driver = AsyncGraphDatabase.driver(
        neo4j_config["uri"],
        auth=(neo4j_config["user"], neo4j_config["password"])
    )
    
    # Clean before test
    async with driver.session(database=neo4j_config["database"]) as session:
        await session.run("MATCH (n) DETACH DELETE n")
    
    yield driver
    
    # Clean after test
    async with driver.session(database=neo4j_config["database"]) as session:
        await session.run("MATCH (n) DETACH DELETE n")
    
    await driver.close()


# ============================================================================
# TEST: Incremental Initialization (test-integration-001)
# ============================================================================

@pytest.mark.integration
class TestIncrementalInitialization:
    """Test incremental initialization behavior."""
    
    @pytest.mark.asyncio
    async def test_full_initialization_on_first_run(
        self, temp_repository, neo4j_config, clean_neo4j
    ):
        """Test that first initialization performs full indexing."""
        # Arrange
        initializer = RepositoryInitializer(
            neo4j_uri=neo4j_config["uri"],
            PROJECT_WATCH_USER=neo4j_config["user"],
            PROJECT_WATCH_PASSWORD=neo4j_config["password"],
            PROJECT_WATCH_DATABASE=neo4j_config["database"],
            repository_path=temp_repository,
            project_name="test_project"
        )
        
        # Act
        start_time = time.time()
        async with initializer:
            result = await initializer.initialize(persistent_monitoring=False)
        full_index_time = time.time() - start_time
        
        # Assert
        assert result.indexed == 4  # All 4 files should be indexed
        assert result.total == 4
        assert len(result.skipped) == 0
        
        # Verify files are in Neo4j
        async with clean_neo4j.session(database=neo4j_config["database"]) as session:
            count_result = await session.run(
                "MATCH (f:CodeFile {project_name: $project}) RETURN count(f) as count",
                project="test_project"
            )
            record = await count_result.single()
            assert record["count"] == 4
        
        return full_index_time
    
    @pytest.mark.asyncio
    async def test_incremental_on_second_run(
        self, temp_repository, neo4j_config, clean_neo4j
    ):
        """Test that second initialization uses incremental indexing."""
        # First initialization
        initializer = RepositoryInitializer(
            neo4j_uri=neo4j_config["uri"],
            PROJECT_WATCH_USER=neo4j_config["user"],
            PROJECT_WATCH_PASSWORD=neo4j_config["password"],
            PROJECT_WATCH_DATABASE=neo4j_config["database"],
            repository_path=temp_repository,
            project_name="test_project"
        )
        
        async with initializer:
            first_result = await initializer.initialize(persistent_monitoring=False)
        
        assert first_result.indexed == 4
        
        # Second initialization (no changes)
        start_time = time.time()
        async with initializer:
            second_result = await initializer.initialize(persistent_monitoring=False)
        incremental_time = time.time() - start_time
        
        # Assert
        assert second_result.indexed == 0  # No files should be re-indexed
        assert second_result.total == 4
        
        # Verify performance improvement
        # Incremental should be significantly faster (we can't compare times without first run time)
        assert incremental_time < 2.0  # Should complete quickly
        
    @pytest.mark.asyncio
    async def test_performance_comparison(
        self, temp_repository, neo4j_config, clean_neo4j
    ):
        """Test performance improvement of incremental vs full indexing."""
        # First run - full indexing
        initializer = RepositoryInitializer(
            neo4j_uri=neo4j_config["uri"],
            PROJECT_WATCH_USER=neo4j_config["user"],
            PROJECT_WATCH_PASSWORD=neo4j_config["password"],
            PROJECT_WATCH_DATABASE=neo4j_config["database"],
            repository_path=temp_repository,
            project_name="test_project"
        )
        
        start_time = time.time()
        async with initializer:
            await initializer.initialize(persistent_monitoring=False)
        full_index_time = time.time() - start_time
        
        # Second run - incremental (no changes)
        start_time = time.time()
        async with initializer:
            await initializer.initialize(persistent_monitoring=False)
        incremental_time = time.time() - start_time
        
        # Assert significant performance improvement
        # Incremental should be at least 50% faster
        improvement = (full_index_time - incremental_time) / full_index_time
        assert improvement > 0.5, f"Expected >50% improvement, got {improvement*100:.1f}%"


# ============================================================================
# TEST: Re-initialization with Changes (test-integration-002)
# ============================================================================

@pytest.mark.integration
class TestReInitializationWithChanges:
    """Test re-initialization only indexes changed files."""
    
    @pytest.mark.asyncio
    async def test_only_modified_files_reindexed(
        self, temp_repository, neo4j_config, clean_neo4j
    ):
        """Test that only modified files are re-indexed."""
        # Initial indexing
        initializer = RepositoryInitializer(
            neo4j_uri=neo4j_config["uri"],
            PROJECT_WATCH_USER=neo4j_config["user"],
            PROJECT_WATCH_PASSWORD=neo4j_config["password"],
            PROJECT_WATCH_DATABASE=neo4j_config["database"],
            repository_path=temp_repository,
            project_name="test_project"
        )
        
        async with initializer:
            first_result = await initializer.initialize(persistent_monitoring=False)
        assert first_result.indexed == 4
        
        # Wait a moment to ensure timestamp difference
        await asyncio.sleep(0.1)
        
        # Modify one file
        (temp_repository / "file1.py").write_text("def modified(): pass")
        
        # Touch the file to update its timestamp
        os.utime(temp_repository / "file1.py", None)
        
        # Re-initialize
        async with initializer:
            second_result = await initializer.initialize(persistent_monitoring=False)
        
        # Assert only the modified file was re-indexed
        assert second_result.indexed == 1
        assert second_result.total == 4
    
    @pytest.mark.asyncio
    async def test_new_files_added(
        self, temp_repository, neo4j_config, clean_neo4j
    ):
        """Test that new files are properly indexed."""
        # Initial indexing
        initializer = RepositoryInitializer(
            neo4j_uri=neo4j_config["uri"],
            PROJECT_WATCH_USER=neo4j_config["user"],
            PROJECT_WATCH_PASSWORD=neo4j_config["password"],
            PROJECT_WATCH_DATABASE=neo4j_config["database"],
            repository_path=temp_repository,
            project_name="test_project"
        )
        
        async with initializer:
            first_result = await initializer.initialize(persistent_monitoring=False)
        assert first_result.indexed == 4
        
        # Add new files
        (temp_repository / "new_file.py").write_text("def new_func(): pass")
        (temp_repository / "another_new.js").write_text("function another() {}")
        
        # Re-initialize
        async with initializer:
            second_result = await initializer.initialize(persistent_monitoring=False)
        
        # Assert new files were indexed
        assert second_result.indexed == 2
        assert second_result.total == 6
        
        # Verify in Neo4j
        async with clean_neo4j.session(database=neo4j_config["database"]) as session:
            count_result = await session.run(
                "MATCH (f:CodeFile {project_name: $project}) RETURN count(f) as count",
                project="test_project"
            )
            record = await count_result.single()
            assert record["count"] == 6
    
    @pytest.mark.asyncio
    async def test_deleted_files_removed(
        self, temp_repository, neo4j_config, clean_neo4j
    ):
        """Test that deleted files are removed from index."""
        # Initial indexing
        initializer = RepositoryInitializer(
            neo4j_uri=neo4j_config["uri"],
            PROJECT_WATCH_USER=neo4j_config["user"],
            PROJECT_WATCH_PASSWORD=neo4j_config["password"],
            PROJECT_WATCH_DATABASE=neo4j_config["database"],
            repository_path=temp_repository,
            project_name="test_project"
        )
        
        async with initializer:
            first_result = await initializer.initialize(persistent_monitoring=False)
        assert first_result.indexed == 4
        
        # Delete a file
        (temp_repository / "file2.py").unlink()
        
        # Re-initialize
        async with initializer:
            second_result = await initializer.initialize(persistent_monitoring=False)
        
        # Assert proper handling
        assert second_result.indexed == 0  # No new indexing
        assert second_result.total == 3    # One less file
        
        # Verify in Neo4j
        async with clean_neo4j.session(database=neo4j_config["database"]) as session:
            count_result = await session.run(
                "MATCH (f:CodeFile {project_name: $project}) RETURN count(f) as count",
                project="test_project"
            )
            record = await count_result.single()
            assert record["count"] == 3
            
            # Verify specific file was deleted
            deleted_result = await session.run(
                "MATCH (f:CodeFile {project_name: $project, path: $path}) RETURN f",
                project="test_project",
                path=str(temp_repository / "file2.py")
            )
            records = [r async for r in deleted_result]
            assert len(records) == 0


# ============================================================================
# TEST: Skip Unchanged Files (test-integration-004)
# ============================================================================

@pytest.mark.integration
class TestSkipUnchangedFiles:
    """Test that unchanged files are not re-indexed on restart."""
    
    @pytest.mark.asyncio
    async def test_no_reindexing_when_no_changes(
        self, temp_repository, neo4j_config, clean_neo4j
    ):
        """Verify no re-indexing occurs when repository hasn't changed."""
        initializer = RepositoryInitializer(
            neo4j_uri=neo4j_config["uri"],
            PROJECT_WATCH_USER=neo4j_config["user"],
            PROJECT_WATCH_PASSWORD=neo4j_config["password"],
            PROJECT_WATCH_DATABASE=neo4j_config["database"],
            repository_path=temp_repository,
            project_name="test_project"
        )
        
        # First initialization
        async with initializer:
            first_result = await initializer.initialize(persistent_monitoring=False)
        
        # Track indexing operations
        with patch.object(Neo4jRAG, 'index_file', new_callable=AsyncMock) as mock_index:
            mock_index.return_value = None
            
            # Second initialization without changes
            async with initializer:
                second_result = await initializer.initialize(persistent_monitoring=False)
            
            # Verify no files were re-indexed
            mock_index.assert_not_called()
        
        assert second_result.indexed == 0
        assert second_result.total == 4


# ============================================================================
# TEST: Performance Benchmarks (test-performance-001)
# ============================================================================

@pytest.mark.integration
@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Benchmark incremental vs full indexing performance."""
    
    @pytest.mark.asyncio
    async def test_large_repository_performance(
        self, neo4j_config, clean_neo4j
    ):
        """Test performance with a large number of files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Create many files (simulate large repository)
            num_files = 100
            for i in range(num_files):
                file_path = repo_path / f"file_{i}.py"
                file_path.write_text(f"def function_{i}(): pass")
            
            initializer = RepositoryInitializer(
                neo4j_uri=neo4j_config["uri"],
                PROJECT_WATCH_USER=neo4j_config["user"],
                PROJECT_WATCH_PASSWORD=neo4j_config["password"],
                PROJECT_WATCH_DATABASE=neo4j_config["database"],
                repository_path=repo_path,
                project_name="test_large_project"
            )
            
            # First run - full indexing
            start_time = time.time()
            async with initializer:
                first_result = await initializer.initialize(persistent_monitoring=False)
            full_index_time = time.time() - start_time
            
            assert first_result.indexed == num_files
            
            # Modify 10% of files
            num_modified = num_files // 10
            for i in range(num_modified):
                file_path = repo_path / f"file_{i}.py"
                file_path.write_text(f"def modified_function_{i}(): pass")
                os.utime(file_path, None)  # Update timestamp
            
            # Second run - incremental indexing
            start_time = time.time()
            async with initializer:
                second_result = await initializer.initialize(persistent_monitoring=False)
            incremental_time = time.time() - start_time
            
            assert second_result.indexed == num_modified
            
            # Performance assertions
            improvement = (full_index_time - incremental_time) / full_index_time
            assert improvement > 0.5, f"Expected >50% improvement, got {improvement*100:.1f}%"
            
            # For 10% changes, should be roughly 80%+ faster
            assert improvement > 0.8, f"Expected >80% improvement for 10% changes, got {improvement*100:.1f}%"


# ============================================================================
# TEST: Edge Cases
# ============================================================================

@pytest.mark.integration
class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_corrupted_index_recovery(
        self, temp_repository, neo4j_config, clean_neo4j
    ):
        """Test graceful fallback when index is corrupted."""
        # Create initial index
        initializer = RepositoryInitializer(
            neo4j_uri=neo4j_config["uri"],
            PROJECT_WATCH_USER=neo4j_config["user"],
            PROJECT_WATCH_PASSWORD=neo4j_config["password"],
            PROJECT_WATCH_DATABASE=neo4j_config["database"],
            repository_path=temp_repository,
            project_name="test_project"
        )
        
        async with initializer:
            await initializer.initialize(persistent_monitoring=False)
        
        # Corrupt the index by removing some relationships
        async with clean_neo4j.session(database=neo4j_config["database"]) as session:
            await session.run("MATCH ()-[r:HAS_CHUNK]->() DELETE r")
        
        # Re-initialize should handle corruption gracefully
        async with initializer:
            result = await initializer.initialize(persistent_monitoring=False)
        
        # Should still work (might re-index everything or handle gracefully)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_timestamp_edge_cases(
        self, temp_repository, neo4j_config, clean_neo4j
    ):
        """Test handling of various timestamp scenarios."""
        initializer = RepositoryInitializer(
            neo4j_uri=neo4j_config["uri"],
            PROJECT_WATCH_USER=neo4j_config["user"],
            PROJECT_WATCH_PASSWORD=neo4j_config["password"],
            PROJECT_WATCH_DATABASE=neo4j_config["database"],
            repository_path=temp_repository,
            project_name="test_project"
        )
        
        # Initial indexing
        async with initializer:
            await initializer.initialize(persistent_monitoring=False)
        
        # Set file timestamp to future
        future_time = time.time() + 3600  # 1 hour in future
        os.utime(temp_repository / "file1.py", (future_time, future_time))
        
        # Should detect as modified
        async with initializer:
            result = await initializer.initialize(persistent_monitoring=False)
        
        assert result.indexed == 1  # Future timestamp treated as modification
        
        # Set file timestamp to very old
        old_time = 0  # Unix epoch
        os.utime(temp_repository / "file2.py", (old_time, old_time))
        
        # Should not re-index (old timestamp means no change)
        async with initializer:
            result = await initializer.initialize(persistent_monitoring=False)
        
        assert result.indexed == 0  # Old timestamp not treated as modification
    
    @pytest.mark.asyncio
    async def test_concurrent_modifications(
        self, temp_repository, neo4j_config, clean_neo4j
    ):
        """Test handling of files modified during indexing."""
        initializer = RepositoryInitializer(
            neo4j_uri=neo4j_config["uri"],
            PROJECT_WATCH_USER=neo4j_config["user"],
            PROJECT_WATCH_PASSWORD=neo4j_config["password"],
            PROJECT_WATCH_DATABASE=neo4j_config["database"],
            repository_path=temp_repository,
            project_name="test_project"
        )
        
        # Initial indexing
        async with initializer:
            await initializer.initialize(persistent_monitoring=False)
        
        # Simulate concurrent modification during indexing
        async def modify_during_indexing():
            await asyncio.sleep(0.01)  # Small delay
            (temp_repository / "file1.py").write_text("def concurrent_mod(): pass")
        
        # Start modification task and indexing concurrently
        with patch.object(Neo4jRAG, 'index_file', new_callable=AsyncMock) as mock_index:
            # Slow down indexing to allow concurrent modification
            async def slow_index(*args, **kwargs):
                await asyncio.sleep(0.02)
                return None
            
            mock_index.side_effect = slow_index
            
            # Run both concurrently
            modify_task = asyncio.create_task(modify_during_indexing())
            
            async with initializer:
                result = await initializer.initialize(persistent_monitoring=False)
            
            await modify_task
        
        # System should handle this gracefully
        assert result is not None


# ============================================================================
# TEST: Multi-Project Isolation (test-e2e-002)
# ============================================================================

@pytest.mark.integration
class TestMultiProjectIsolation:
    """Test that multiple projects maintain separate incremental indexes."""
    
    @pytest.mark.asyncio
    async def test_multiple_projects_isolated(
        self, neo4j_config, clean_neo4j
    ):
        """Verify incremental indexing maintains project isolation."""
        with tempfile.TemporaryDirectory() as tmpdir1, \
             tempfile.TemporaryDirectory() as tmpdir2:
            
            repo1 = Path(tmpdir1)
            repo2 = Path(tmpdir2)
            
            # Create files in both repos
            (repo1 / "project1_file.py").write_text("def project1(): pass")
            (repo2 / "project2_file.py").write_text("def project2(): pass")
            
            # Initialize first project
            init1 = RepositoryInitializer(
                neo4j_uri=neo4j_config["uri"],
                PROJECT_WATCH_USER=neo4j_config["user"],
                PROJECT_WATCH_PASSWORD=neo4j_config["password"],
                PROJECT_WATCH_DATABASE=neo4j_config["database"],
                repository_path=repo1,
                project_name="project1"
            )
            
            async with init1:
                result1 = await init1.initialize(persistent_monitoring=False)
            assert result1.indexed == 1
            
            # Initialize second project
            init2 = RepositoryInitializer(
                neo4j_uri=neo4j_config["uri"],
                PROJECT_WATCH_USER=neo4j_config["user"],
                PROJECT_WATCH_PASSWORD=neo4j_config["password"],
                PROJECT_WATCH_DATABASE=neo4j_config["database"],
                repository_path=repo2,
                project_name="project2"
            )
            
            async with init2:
                result2 = await init2.initialize(persistent_monitoring=False)
            assert result2.indexed == 1
            
            # Re-initialize first project (should use incremental)
            async with init1:
                result1_again = await init1.initialize(persistent_monitoring=False)
            assert result1_again.indexed == 0  # No changes
            
            # Modify second project
            (repo2 / "new_file.py").write_text("def new(): pass")
            
            # Re-initialize second project
            async with init2:
                result2_again = await init2.initialize(persistent_monitoring=False)
            assert result2_again.indexed == 1  # Only new file
            
            # Verify isolation in Neo4j
            async with clean_neo4j.session(database=neo4j_config["database"]) as session:
                # Check project1 files
                p1_result = await session.run(
                    "MATCH (f:CodeFile {project_name: $project}) RETURN count(f) as count",
                    project="project1"
                )
                p1_record = await p1_result.single()
                assert p1_record["count"] == 1
                
                # Check project2 files
                p2_result = await session.run(
                    "MATCH (f:CodeFile {project_name: $project}) RETURN count(f) as count",
                    project="project2"
                )
                p2_record = await p2_result.single()
                assert p2_record["count"] == 2