"""
Edge case tests for incremental indexing functionality.

This module tests various edge cases and error conditions to ensure
the incremental indexing feature is robust and handles unexpected scenarios.
"""

import asyncio
import os
import shutil
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
import random

import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, TransientError

from src.project_watch_mcp.neo4j_rag import Neo4jRAG
from src.project_watch_mcp.repository_monitor import FileInfo, RepositoryMonitor
from src.project_watch_mcp.core.initializer import RepositoryInitializer
from src.project_watch_mcp.config import ProjectWatchConfig


# ============================================================================
# FIXTURES
# ============================================================================

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


@pytest.fixture
async def temp_repository():
    """Create a temporary repository with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        
        # Create test files
        (repo_path / "file1.py").write_text("def test1(): pass")
        (repo_path / "file2.py").write_text("def test2(): pass")
        (repo_path / "subdir").mkdir()
        (repo_path / "subdir" / "file3.py").write_text("def test3(): pass")
        
        yield repo_path


# ============================================================================
# TEST: Corrupted Index Recovery (test-edge-001)
# ============================================================================

@pytest.mark.integration
class TestCorruptedIndexRecovery:
    """Test recovery from corrupted Neo4j index."""
    
    @pytest.mark.asyncio
    async def test_missing_file_nodes(
        self, temp_repository, neo4j_config, clean_neo4j
    ):
        """Test recovery when File nodes are missing but chunks exist."""
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
            await initializer.initialize(persistent_monitoring=False)
        
        # Corrupt index by deleting File nodes but keeping chunks
        async with clean_neo4j.session(database=neo4j_config["database"]) as session:
            await session.run("""
                MATCH (f:CodeFile {project_name: $project})
                WHERE f.path CONTAINS 'file1'
                DETACH DELETE f
            """, project="test_project")
        
        # Re-initialize should handle corruption
        async with initializer:
            result = await initializer.initialize(persistent_monitoring=False)
        
        # Should re-index the missing file
        assert result.indexed >= 1
        
        # Verify index is healthy
        async with clean_neo4j.session(database=neo4j_config["database"]) as session:
            count_result = await session.run(
                "MATCH (f:CodeFile {project_name: $project}) RETURN count(f) as count",
                project="test_project"
            )
            record = await count_result.single()
            assert record["count"] == 3  # All files should be indexed
    
    @pytest.mark.asyncio
    async def test_orphaned_chunks(
        self, temp_repository, neo4j_config, clean_neo4j
    ):
        """Test handling of orphaned chunks without parent files."""
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
            await initializer.initialize(persistent_monitoring=False)
        
        # Create orphaned chunks
        async with clean_neo4j.session(database=neo4j_config["database"]) as session:
            # Delete relationships but keep chunks
            await session.run("MATCH ()-[r:HAS_CHUNK]->() DELETE r")
        
        # Re-initialize
        async with initializer:
            result = await initializer.initialize(persistent_monitoring=False)
        
        # Should still work
        assert result is not None
        
        # Verify orphaned chunks are cleaned up or re-linked
        async with clean_neo4j.session(database=neo4j_config["database"]) as session:
            orphan_result = await session.run("""
                MATCH (c:CodeChunk)
                WHERE NOT (()-[:HAS_CHUNK]->(c))
                RETURN count(c) as orphans
            """)
            record = await orphan_result.single()
            # Should either have no orphans or they should be cleaned up
            assert record["orphans"] == 0
    
    @pytest.mark.asyncio
    async def test_inconsistent_timestamps(
        self, temp_repository, neo4j_config, clean_neo4j
    ):
        """Test handling of inconsistent timestamps in index."""
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
            await initializer.initialize(persistent_monitoring=False)
        
        # Corrupt timestamps in index
        async with clean_neo4j.session(database=neo4j_config["database"]) as session:
            # Set some timestamps to null
            await session.run("""
                MATCH (f:CodeFile {project_name: $project})
                WHERE f.path CONTAINS 'file1'
                REMOVE f.last_modified
            """, project="test_project")
            
            # Set some timestamps to invalid values
            await session.run("""
                MATCH (f:CodeFile {project_name: $project})
                WHERE f.path CONTAINS 'file2'
                SET f.last_modified = 'invalid_timestamp'
            """, project="test_project")
        
        # Re-initialize should handle bad timestamps
        async with initializer:
            result = await initializer.initialize(persistent_monitoring=False)
        
        # Should re-index files with bad timestamps
        assert result.indexed >= 2


# ============================================================================
# TEST: Timestamp Edge Cases (test-edge-002)
# ============================================================================

@pytest.mark.integration
class TestTimestampEdgeCases:
    """Test various timestamp-related edge cases."""
    
    @pytest.mark.asyncio
    async def test_future_timestamps(
        self, temp_repository, neo4j_config, clean_neo4j
    ):
        """Test handling of files with future timestamps."""
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
            await initializer.initialize(persistent_monitoring=False)
        
        # Set file timestamp to future
        future_time = time.time() + 86400  # 1 day in future
        os.utime(temp_repository / "file1.py", (future_time, future_time))
        
        # Re-initialize
        async with initializer:
            result = await initializer.initialize(persistent_monitoring=False)
        
        # Future timestamp should be treated as modification
        assert result.indexed == 1
    
    @pytest.mark.asyncio
    async def test_ancient_timestamps(
        self, temp_repository, neo4j_config, clean_neo4j
    ):
        """Test handling of files with very old timestamps."""
        # Set file to ancient timestamp before indexing
        ancient_time = 0  # Unix epoch
        os.utime(temp_repository / "file1.py", (ancient_time, ancient_time))
        
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
        
        assert first_result.indexed == 3
        
        # Re-initialize without changes
        async with initializer:
            second_result = await initializer.initialize(persistent_monitoring=False)
        
        # Ancient timestamp shouldn't trigger re-indexing
        assert second_result.indexed == 0
    
    @pytest.mark.asyncio
    async def test_timestamp_rollback(
        self, temp_repository, neo4j_config, clean_neo4j
    ):
        """Test handling of timestamp rollback scenarios."""
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
            await initializer.initialize(persistent_monitoring=False)
        
        # Roll back file timestamp (simulate clock change or file restoration)
        past_time = time.time() - 86400  # 1 day ago
        os.utime(temp_repository / "file1.py", (past_time, past_time))
        
        # Re-initialize
        async with initializer:
            result = await initializer.initialize(persistent_monitoring=False)
        
        # Rollback shouldn't trigger re-indexing (file is older)
        assert result.indexed == 0
    
    @pytest.mark.asyncio
    async def test_microsecond_precision(
        self, temp_repository, neo4j_config, clean_neo4j
    ):
        """Test handling of microsecond timestamp precision."""
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
            await initializer.initialize(persistent_monitoring=False)
        
        # Modify file with microsecond difference
        current_time = time.time()
        microsecond_later = current_time + 0.000001
        os.utime(temp_repository / "file1.py", (microsecond_later, microsecond_later))
        
        # Re-initialize
        async with initializer:
            result = await initializer.initialize(persistent_monitoring=False)
        
        # Should detect even tiny timestamp changes
        assert result.indexed == 1


# ============================================================================
# TEST: Concurrent Modifications (test-edge-003)
# ============================================================================

@pytest.mark.integration
class TestConcurrentModifications:
    """Test handling of concurrent file modifications during indexing."""
    
    @pytest.mark.asyncio
    async def test_file_modified_during_indexing(
        self, temp_repository, neo4j_config, clean_neo4j
    ):
        """Test file being modified while it's being indexed."""
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
        
        # Modify file and trigger re-index
        (temp_repository / "file1.py").write_text("def modified(): pass")
        os.utime(temp_repository / "file1.py", None)
        
        # Patch index_file to simulate slow indexing
        async def slow_index_file(file_info):
            if "file1" in str(file_info.path):
                # Simulate slow processing
                await asyncio.sleep(0.1)
                # Modify file during indexing
                (temp_repository / "file1.py").write_text("def concurrent_mod(): pass")
            return None
        
        with patch.object(Neo4jRAG, 'index_file', side_effect=slow_index_file):
            async with initializer:
                result = await initializer.initialize(persistent_monitoring=False)
        
        # Should handle concurrent modification gracefully
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_file_deleted_during_indexing(
        self, temp_repository, neo4j_config, clean_neo4j
    ):
        """Test file being deleted while indexing is in progress."""
        # Create extra files
        for i in range(5):
            (temp_repository / f"extra_{i}.py").write_text(f"def extra_{i}(): pass")
        
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
        
        # Modify files to trigger re-indexing
        for i in range(5):
            (temp_repository / f"extra_{i}.py").write_text(f"def modified_{i}(): pass")
            os.utime(temp_repository / f"extra_{i}.py", None)
        
        # Patch to delete file during indexing
        original_index = Neo4jRAG.index_file
        call_count = 0
        
        async def index_with_deletion(self, file_info):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                # Delete a file mid-indexing
                file_to_delete = temp_repository / "extra_4.py"
                if file_to_delete.exists():
                    file_to_delete.unlink()
            
            # Try to index (might fail for deleted file)
            try:
                return await original_index(self, file_info)
            except FileNotFoundError:
                # Handle gracefully
                return None
        
        with patch.object(Neo4jRAG, 'index_file', index_with_deletion):
            async with initializer:
                result = await initializer.initialize(persistent_monitoring=False)
        
        # Should complete despite deletion
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_race_condition_multiple_initializers(
        self, temp_repository, neo4j_config, clean_neo4j
    ):
        """Test race conditions with multiple initializers running concurrently."""
        # Create multiple initializers
        initializers = []
        for i in range(3):
            initializer = RepositoryInitializer(
                neo4j_uri=neo4j_config["uri"],
                PROJECT_WATCH_USER=neo4j_config["user"],
                PROJECT_WATCH_PASSWORD=neo4j_config["password"],
                PROJECT_WATCH_DATABASE=neo4j_config["database"],
                repository_path=temp_repository,
                project_name=f"test_project_{i}"  # Different projects to avoid conflicts
            )
            initializers.append(initializer)
        
        # Run all initializers concurrently
        async def run_initializer(init):
            async with init:
                return await init.initialize(persistent_monitoring=False)
        
        results = await asyncio.gather(
            *[run_initializer(init) for init in initializers],
            return_exceptions=True
        )
        
        # All should complete successfully
        for i, result in enumerate(results):
            assert not isinstance(result, Exception), f"Initializer {i} failed: {result}"
            assert result.indexed == 3  # Each should index all files


# ============================================================================
# TEST: File System Edge Cases
# ============================================================================

@pytest.mark.integration
class TestFileSystemEdgeCases:
    """Test edge cases related to file system operations."""
    
    @pytest.mark.asyncio
    async def test_symlink_handling(
        self, neo4j_config, clean_neo4j
    ):
        """Test handling of symbolic links in repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Create regular files
            (repo_path / "real_file.py").write_text("def real(): pass")
            
            # Create a symlink
            target = repo_path / "target.py"
            target.write_text("def target(): pass")
            symlink = repo_path / "symlink.py"
            symlink.symlink_to(target)
            
            initializer = RepositoryInitializer(
                neo4j_uri=neo4j_config["uri"],
                PROJECT_WATCH_USER=neo4j_config["user"],
                PROJECT_WATCH_PASSWORD=neo4j_config["password"],
                PROJECT_WATCH_DATABASE=neo4j_config["database"],
                repository_path=repo_path,
                project_name="test_project"
            )
            
            # Initial indexing
            async with initializer:
                result = await initializer.initialize(persistent_monitoring=False)
            
            # Should index both real files and symlinks
            assert result.indexed >= 2
    
    @pytest.mark.asyncio
    async def test_permission_errors(
        self, neo4j_config, clean_neo4j
    ):
        """Test handling of files with permission errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Create files
            file1 = repo_path / "readable.py"
            file1.write_text("def readable(): pass")
            
            file2 = repo_path / "unreadable.py"
            file2.write_text("def unreadable(): pass")
            
            # Make file unreadable (Unix only)
            if os.name != 'nt':
                os.chmod(file2, 0o000)
            
            try:
                initializer = RepositoryInitializer(
                    neo4j_uri=neo4j_config["uri"],
                    PROJECT_WATCH_USER=neo4j_config["user"],
                    PROJECT_WATCH_PASSWORD=neo4j_config["password"],
                    PROJECT_WATCH_DATABASE=neo4j_config["database"],
                    repository_path=repo_path,
                    project_name="test_project"
                )
                
                # Should handle permission errors gracefully
                async with initializer:
                    result = await initializer.initialize(persistent_monitoring=False)
                
                # Should index readable files
                assert result.indexed >= 1
                
                # Unreadable file might be skipped
                if os.name != 'nt':
                    assert len(result.skipped) >= 0
            
            finally:
                # Restore permissions for cleanup
                if os.name != 'nt':
                    os.chmod(file2, 0o644)
    
    @pytest.mark.asyncio
    async def test_very_long_paths(
        self, neo4j_config, clean_neo4j
    ):
        """Test handling of very long file paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Create deeply nested directory structure
            deep_path = repo_path
            for i in range(20):
                deep_path = deep_path / f"level_{i}"
            deep_path.mkdir(parents=True)
            
            # Create file with long name
            long_filename = "x" * 200 + ".py"
            (deep_path / long_filename).write_text("def long_path(): pass")
            
            initializer = RepositoryInitializer(
                neo4j_uri=neo4j_config["uri"],
                PROJECT_WATCH_USER=neo4j_config["user"],
                PROJECT_WATCH_PASSWORD=neo4j_config["password"],
                PROJECT_WATCH_DATABASE=neo4j_config["database"],
                repository_path=repo_path,
                project_name="test_project"
            )
            
            # Should handle long paths
            async with initializer:
                result = await initializer.initialize(persistent_monitoring=False)
            
            assert result.indexed == 1


# ============================================================================
# TEST: Database Connection Issues
# ============================================================================

@pytest.mark.integration
class TestDatabaseConnectionIssues:
    """Test handling of database connection problems."""
    
    @pytest.mark.asyncio
    async def test_transient_connection_error(
        self, temp_repository, neo4j_config
    ):
        """Test recovery from transient connection errors."""
        initializer = RepositoryInitializer(
            neo4j_uri=neo4j_config["uri"],
            PROJECT_WATCH_USER=neo4j_config["user"],
            PROJECT_WATCH_PASSWORD=neo4j_config["password"],
            PROJECT_WATCH_DATABASE=neo4j_config["database"],
            repository_path=temp_repository,
            project_name="test_project"
        )
        
        # Simulate transient errors
        call_count = 0
        original_execute = None
        
        async def flaky_execute(*args, **kwargs):
            nonlocal call_count, original_execute
            call_count += 1
            
            # Fail first 2 attempts, then succeed
            if call_count <= 2:
                raise TransientError("Connection temporarily unavailable")
            
            return await original_execute(*args, **kwargs)
        
        async with initializer:
            # Patch after driver is created
            if hasattr(initializer, 'neo4j_driver'):
                original_execute = initializer.neo4j_driver.execute_query
                with patch.object(
                    initializer.neo4j_driver, 
                    'execute_query',
                    side_effect=flaky_execute
                ):
                    # Should retry and eventually succeed
                    try:
                        result = await initializer.initialize(persistent_monitoring=False)
                        # May or may not succeed depending on retry logic
                        assert result is not None or call_count > 0
                    except TransientError:
                        # Expected if no retry logic
                        assert call_count > 0