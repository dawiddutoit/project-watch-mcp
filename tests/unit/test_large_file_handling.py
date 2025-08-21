"""Test handling of large files and chunks to prevent Lucene index failures."""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neo4j import AsyncDriver, RoutingControl

from project_watch_mcp.neo4j_rag import CodeFile, Neo4jRAG
from project_watch_mcp.repository_monitor import RepositoryMonitor


@pytest.fixture
def mock_driver():
    """Create a mock Neo4j driver."""
    driver = MagicMock(spec=AsyncDriver)
    driver.execute_query = AsyncMock()
    return driver


@pytest.fixture
def neo4j_rag(mock_driver):
    """Create a Neo4jRAG instance with mocked driver."""
    return Neo4jRAG(
        neo4j_driver=mock_driver,
        project_name="test_project",
        embeddings=None,
        chunk_size=100,  # Small chunk size for testing
        chunk_overlap=20
    )


@pytest.fixture
def repository_monitor(mock_driver):
    """Create a RepositoryMonitor instance."""
    return RepositoryMonitor(
        repo_path=Path("/test/repo"),
        project_name="test_project",
        neo4j_driver=mock_driver,
        use_gitignore=False
    )


class TestLockFileExclusion:
    """Test that lock files are excluded from indexing."""
    
    def test_uv_lock_excluded(self, repository_monitor):
        """Test that uv.lock files are excluded."""
        assert not repository_monitor._should_include_file(Path("/test/repo/uv.lock"))
    
    def test_package_lock_json_excluded(self, repository_monitor):
        """Test that package-lock.json files are excluded."""
        assert not repository_monitor._should_include_file(Path("/test/repo/package-lock.json"))
    
    def test_yarn_lock_excluded(self, repository_monitor):
        """Test that yarn.lock files are excluded."""
        assert not repository_monitor._should_include_file(Path("/test/repo/yarn.lock"))
    
    def test_poetry_lock_excluded(self, repository_monitor):
        """Test that poetry.lock files are excluded."""
        assert not repository_monitor._should_include_file(Path("/test/repo/poetry.lock"))
    
    def test_pipfile_lock_excluded(self, repository_monitor):
        """Test that Pipfile.lock files are excluded."""
        assert not repository_monitor._should_include_file(Path("/test/repo/Pipfile.lock"))
    
    def test_generic_lock_files_excluded(self, repository_monitor):
        """Test that any .lock file is excluded."""
        assert not repository_monitor._should_include_file(Path("/test/repo/some.lock"))
        assert not repository_monitor._should_include_file(Path("/test/repo/custom.lock"))
    
    def test_normal_files_included(self, repository_monitor):
        """Test that normal files are still included."""
        # Python files should be included
        assert repository_monitor._should_include_file(Path("/test/repo/main.py"))
        # JavaScript files should be included
        assert repository_monitor._should_include_file(Path("/test/repo/app.js"))
        # README should be included
        assert repository_monitor._should_include_file(Path("/test/repo/README.md"))


class TestLargeChunkHandling:
    """Test that chunks exceeding 32KB are handled properly."""
    
    @pytest.mark.asyncio
    async def test_large_chunk_skipped_with_warning(self, neo4j_rag, mock_driver, caplog):
        """Test that chunks larger than 32KB are skipped with a warning."""
        # Create content that will result in a chunk larger than 32KB
        large_content = "x" * 35000  # 35KB of 'x' characters
        
        code_file = CodeFile(
            project_name="test_project",
            path=Path("/test/large_file.txt"),
            content=large_content,
            language="text",
            size=len(large_content),
            last_modified=datetime.now()
        )
        
        # Index the file
        with caplog.at_level("WARNING"):
            await neo4j_rag.index_file(code_file)
        
        # Check that a warning was logged
        assert "exceeds 32KB limit" in caplog.text
        assert "large_file.txt" in caplog.text
        
        # Verify that the chunk creation query was NOT called for the large chunk
        calls = mock_driver.execute_query.call_args_list
        chunk_create_calls = [call for call in calls if "CREATE (c:CodeChunk" in call[0][0]]
        
        # Should have no chunk creation calls since the single chunk was too large
        assert len(chunk_create_calls) == 0
    
    @pytest.mark.asyncio
    async def test_mixed_chunks_only_large_skipped(self, neo4j_rag, mock_driver, caplog):
        """Test that only large chunks are skipped, not all chunks."""
        # Create content with mixed chunk sizes
        # First part will create a normal chunk, second part will be too large
        normal_part = "normal content\n" * 50  # Small enough chunk
        large_part = "x" * 35000  # This will create a >32KB chunk
        
        # Adjust neo4j_rag to have larger chunk size for this test
        neo4j_rag.chunk_size = 100000  # Large chunk size to not split the large part
        
        code_file = CodeFile(
            project_name="test_project",
            path=Path("/test/mixed_file.txt"),
            content=normal_part + large_part,
            language="text",
            size=len(normal_part + large_part),
            last_modified=datetime.now()
        )
        
        # Mock the chunk_content method to return our controlled chunks
        with patch.object(neo4j_rag, 'chunk_content') as mock_chunk:
            mock_chunk.return_value = [normal_part, large_part]
            
            with caplog.at_level("WARNING"):
                await neo4j_rag.index_file(code_file)
        
        # Check that a warning was logged for the large chunk
        assert "exceeds 32KB limit" in caplog.text
        
        # Verify that only the normal chunk was created
        calls = mock_driver.execute_query.call_args_list
        chunk_create_calls = [call for call in calls if "CREATE (c:CodeChunk" in call[0][0]]
        
        # Should have exactly 1 chunk creation (the normal one)
        assert len(chunk_create_calls) == 1
        
        # Verify the created chunk is the normal one
        chunk_params = chunk_create_calls[0][0][1]
        assert chunk_params["content"] == normal_part
    
    @pytest.mark.asyncio
    async def test_utf8_encoding_size_check(self, neo4j_rag, mock_driver):
        """Test that UTF-8 encoding size is properly checked."""
        # Create content with multi-byte UTF-8 characters
        # Each emoji is 4 bytes in UTF-8
        emoji_content = "🎉" * 8000  # 8000 emojis = 32000 bytes in UTF-8
        
        code_file = CodeFile(
            project_name="test_project",
            path=Path("/test/emoji_file.txt"),
            content=emoji_content,
            language="text",
            size=len(emoji_content.encode('utf-8')),
            last_modified=datetime.now()
        )
        
        # Mock chunk_content to return the emoji content as a single chunk
        with patch.object(neo4j_rag, 'chunk_content') as mock_chunk:
            mock_chunk.return_value = [emoji_content]
            
            await neo4j_rag.index_file(code_file)
        
        # Verify the chunk was created (exactly at 32000 bytes, not over)
        calls = mock_driver.execute_query.call_args_list
        chunk_create_calls = [call for call in calls if "CREATE (c:CodeChunk" in call[0][0]]
        
        # Should have 1 chunk creation (32000 bytes is exactly at the limit)
        assert len(chunk_create_calls) == 1
        
        # Now test with one more emoji to exceed the limit
        emoji_content_large = "🎉" * 8001  # 32004 bytes - over the limit
        
        code_file_large = CodeFile(
            project_name="test_project",
            path=Path("/test/emoji_file_large.txt"),
            content=emoji_content_large,
            language="text",
            size=len(emoji_content_large.encode('utf-8')),
            last_modified=datetime.now()
        )
        
        mock_driver.execute_query.reset_mock()
        
        with patch.object(neo4j_rag, 'chunk_content') as mock_chunk:
            mock_chunk.return_value = [emoji_content_large]
            
            with patch('project_watch_mcp.neo4j_rag.logger') as mock_logger:
                await neo4j_rag.index_file(code_file_large)
                
                # Verify warning was logged
                mock_logger.warning.assert_called_once()
                warning_message = mock_logger.warning.call_args[0][0]
                assert "exceeds 32KB limit" in warning_message
                assert "32004 bytes" in warning_message
        
        # Verify no chunk was created for the oversized content
        calls = mock_driver.execute_query.call_args_list
        chunk_create_calls = [call for call in calls if "CREATE (c:CodeChunk" in call[0][0]]
        assert len(chunk_create_calls) == 0


class TestRepositoryMonitorPatterns:
    """Test the default ignore patterns in RepositoryMonitor."""
    
    def test_default_ignore_patterns_include_lock_files(self, repository_monitor):
        """Test that default ignore patterns include lock files."""
        # Check that lock file patterns are in the ignore list
        ignore_patterns = repository_monitor.ignore_patterns
        
        # These should be in the default patterns
        assert "*.lock" in ignore_patterns
        assert "uv.lock" in ignore_patterns
        assert "package-lock.json" in ignore_patterns
        assert "yarn.lock" in ignore_patterns
        assert "poetry.lock" in ignore_patterns
        assert "Pipfile.lock" in ignore_patterns
    
    def test_custom_ignore_patterns_additive(self):
        """Test that custom ignore patterns are additive with defaults."""
        mock_driver = MagicMock(spec=AsyncDriver)
        
        custom_patterns = ["*.custom", "special.file"]
        monitor = RepositoryMonitor(
            repo_path=Path("/test/repo"),
            project_name="test_project",
            neo4j_driver=mock_driver,
            ignore_patterns=custom_patterns,
            use_gitignore=False
        )
        
        # Custom patterns should be combined with defaults
        # Defaults always include lock files for safety
        assert "*.custom" in monitor.ignore_patterns
        assert "special.file" in monitor.ignore_patterns
        assert "*.lock" in monitor.ignore_patterns  # Default pattern
        assert "uv.lock" in monitor.ignore_patterns  # Default pattern
        
        # Test that both custom and default patterns work
        assert not monitor._should_include_file(Path("/test/repo/file.custom"))
        assert not monitor._should_include_file(Path("/test/repo/special.file"))
        assert not monitor._should_include_file(Path("/test/repo/uv.lock"))  # Default exclusion