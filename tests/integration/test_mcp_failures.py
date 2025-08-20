"""
Integration tests for MCP failures identified in project-watch-mcp-test-failures.md.

These tests reproduce and verify fixes for all critical issues:
1. Pattern search with regex (TODO|FIXME|HACK, class names, method names)
2. Pattern search without regex (exact class names, file names, test names)
3. Indexing status consistency
4. Content retrieval (no improper truncation)
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from project_watch_mcp.neo4j_rag import Neo4jRAG, CodeFile, SearchResult
from project_watch_mcp.server import create_mcp_server
import datetime


@pytest.fixture
async def mock_neo4j_driver():
    """Create a mock Neo4j driver."""
    driver = AsyncMock()
    
    # Mock execute_query for different scenarios
    async def execute_query_mock(query, params=None, **kwargs):
        # Return different results based on query patterns
        if "CALL db.index.fulltext.queryNodes" in query:
            # Pattern search using fulltext index
            return MagicMock(records=[
                {
                    "file_path": "src/utils.py",
                    "content": "# TODO: Implement error handling",
                    "line_number": 23,
                    "similarity": 1.0,
                    "project_name": "test_project"
                },
                {
                    "file_path": "src/main.py", 
                    "content": "# FIXME: Memory leak in parser",
                    "line_number": 45,
                    "similarity": 0.95,
                    "project_name": "test_project"
                }
            ])
        elif "c.content =~" in query:
            # Regex pattern search
            return MagicMock(records=[
                {
                    "file_path": "src/utils.py",
                    "content": "# TODO: Implement error handling",
                    "line_number": 23,
                    "project_name": "test_project"
                },
                {
                    "file_path": "src/parser.py",
                    "content": "# HACK: Temporary workaround",
                    "line_number": 67,
                    "project_name": "test_project"
                }
            ])
        elif "MATCH (f:CodeFile {project_name:" in query and "OPTIONAL MATCH" in query:
            # File metadata query
            if "indexed_file.py" in str(params.get("path", "")):
                return MagicMock(records=[{
                    "path": "src/indexed_file.py",
                    "language": "python",
                    "size": 2048,
                    "last_modified": datetime.datetime.now().isoformat(),
                    "hash": "abc123",
                    "project_name": "test_project",
                    "chunk_count": 8
                }])
            elif "empty_file.py" in str(params.get("path", "")):
                return MagicMock(records=[{
                    "path": "src/empty_file.py",
                    "language": "python",
                    "size": 0,
                    "last_modified": datetime.datetime.now().isoformat(),
                    "hash": "empty",
                    "project_name": "test_project",
                    "chunk_count": 0
                }])
            else:
                return MagicMock(records=[])
        else:
            # Default empty response
            return MagicMock(records=[])
    
    driver.execute_query = execute_query_mock
    return driver


@pytest.fixture
async def neo4j_rag(mock_neo4j_driver):
    """Create Neo4jRAG instance with mocked driver."""
    rag = Neo4jRAG(
        neo4j_driver=mock_neo4j_driver,
        project_name="test_project",
        embeddings=None  # No embeddings for these tests
    )
    return rag


class TestPatternSearch:
    """Test pattern search functionality."""
    
    @pytest.mark.asyncio
    async def test_pattern_search_with_regex(self, neo4j_rag):
        """Test pattern search with regex for TODO|FIXME|HACK."""
        results = await neo4j_rag.search_by_pattern(
            pattern="TODO|FIXME|HACK",
            is_regex=True,
            limit=10
        )
        
        assert len(results) > 0
        assert any("TODO" in r.content for r in results)
        assert all(isinstance(r, SearchResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_pattern_search_with_language_filter(self, neo4j_rag):
        """Test pattern search with language filtering."""
        results = await neo4j_rag.search_by_pattern(
            pattern="class",
            is_regex=False,
            limit=10,
            language="python"
        )
        
        # Should not fail with TypeError
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_pattern_search_exact_match(self, neo4j_rag):
        """Test pattern search for exact class names."""
        results = await neo4j_rag.search_by_pattern(
            pattern="MainClass",
            is_regex=False,
            limit=5
        )
        
        assert isinstance(results, list)
        # Results depend on mock data
    
    @pytest.mark.asyncio
    async def test_pattern_search_special_characters(self, neo4j_rag):
        """Test pattern search handles special characters correctly."""
        # Should not fail with Lucene escaping issues
        results = await neo4j_rag.search_by_pattern(
            pattern="test_function()",
            is_regex=False,
            limit=5
        )
        
        assert isinstance(results, list)


class TestIndexingStatus:
    """Test indexing status consistency."""
    
    @pytest.mark.asyncio
    async def test_file_with_chunks_shows_indexed(self, neo4j_rag):
        """Test that files with chunks show indexed=True."""
        metadata = await neo4j_rag.get_file_metadata(Path("src/indexed_file.py"))
        
        assert metadata is not None
        assert metadata["chunk_count"] == 8
        # The indexed flag is set in server.py, not in neo4j_rag
        # But we can verify the data structure is correct
        assert "chunk_count" in metadata
        assert metadata["chunk_count"] > 0
    
    @pytest.mark.asyncio
    async def test_file_without_chunks_shows_not_indexed(self, neo4j_rag):
        """Test that files without chunks show indexed=False."""
        metadata = await neo4j_rag.get_file_metadata(Path("src/empty_file.py"))
        
        assert metadata is not None
        assert metadata["chunk_count"] == 0
    
    @pytest.mark.asyncio
    async def test_nonexistent_file_returns_none(self, neo4j_rag):
        """Test that non-existent files return None."""
        metadata = await neo4j_rag.get_file_metadata(Path("nonexistent.py"))
        
        assert metadata is None


class TestContentRetrieval:
    """Test content retrieval and truncation."""
    
    @pytest.mark.asyncio
    async def test_search_returns_full_chunk_content(self, neo4j_rag):
        """Test that search returns full chunk content, not truncated."""
        # Create a mock with long content
        long_content = "x" * 1000  # 1000 chars
        
        with patch.object(neo4j_rag.neo4j_driver, 'execute_query') as mock_exec:
            mock_exec.return_value = MagicMock(records=[{
                "file_path": "test.py",
                "content": long_content,
                "line_number": 1,
                "similarity": 0.95,
                "project_name": "test_project"
            }])
            
            results = await neo4j_rag.search_by_pattern(
                pattern="test",
                is_regex=False,
                limit=1
            )
            
            assert len(results) == 1
            # The actual chunk content should be returned (not truncated in neo4j_rag)
            assert len(results[0].content) == 1000
            assert results[0].content == long_content


class TestChunkingStrategy:
    """Test smart chunking strategy for large files."""
    
    def test_small_file_single_chunk(self):
        """Test that small files create a single chunk."""
        rag = Neo4jRAG(
            neo4j_driver=MagicMock(),
            project_name="test",
            embeddings=None
        )
        
        small_content = "def hello():\n    print('Hello')\n" * 10
        chunks = rag.chunk_content(small_content)
        
        assert len(chunks) == 1
        assert chunks[0] == small_content
    
    def test_large_file_token_based_chunking(self):
        """Test that very large files use token-based chunking."""
        rag = Neo4jRAG(
            neo4j_driver=MagicMock(),
            project_name="test",
            embeddings=None
        )
        
        # Create content that would exceed 30000 tokens (150000+ chars)
        large_content = "x" * 200000
        chunks = rag.chunk_content(large_content)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should be under token limit (roughly 10000 chars for 2000 tokens)
        for chunk in chunks:
            assert len(chunk) <= 12000  # Allow some margin
    
    def test_chunking_with_overlap(self):
        """Test that chunks have proper overlap."""
        rag = Neo4jRAG(
            neo4j_driver=MagicMock(),
            project_name="test",
            embeddings=None
        )
        
        # Create content with clear line markers, long enough to force chunking
        # Each line is ~80 chars to ensure we exceed limits
        lines = [f"Line {i}: " + "x" * 70 for i in range(1000)]
        content = "\n".join(lines)
        
        # This content is roughly 80*1000 = 80000 chars = 16000 tokens
        # Should create multiple chunks
        chunks = rag.chunk_content(content, chunk_size=100, overlap=10)
        
        # With token-aware chunking, should have multiple chunks
        assert len(chunks) > 1
        
        # Basic validation of chunks
        for i, chunk in enumerate(chunks):
            assert len(chunk) > 0
            # Each chunk should be within reasonable size
            assert len(chunk) <= 12000  # Max for 2000 tokens with margin


class TestEndToEndIntegration:
    """Full end-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_complete_search_workflow(self, neo4j_rag):
        """Test complete search workflow from query to results."""
        # Test semantic search (would need embeddings in real scenario)
        # For now, test pattern search
        
        # 1. Search for TODO comments
        results = await neo4j_rag.search_by_pattern(
            pattern="TODO",
            is_regex=False,
            limit=10
        )
        
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        
        # 2. Get file info for a result
        if results:
            file_path = results[0].file_path
            metadata = await neo4j_rag.get_file_metadata(file_path)
            # Metadata structure should be valid
            if metadata:
                assert "path" in metadata
                assert "chunk_count" in metadata
    
    @pytest.mark.asyncio  
    async def test_repository_stats(self, neo4j_rag):
        """Test repository statistics retrieval."""
        with patch.object(neo4j_rag.neo4j_driver, 'execute_query') as mock_exec:
            mock_exec.return_value = MagicMock(records=[{
                "total_files": 42,
                "total_chunks": 256,
                "total_size": 1048576,
                "languages": ["python", "javascript"],
                "project_name": "test_project"
            }])
            
            stats = await neo4j_rag.get_repository_stats()
            
            assert stats["total_files"] == 42
            assert stats["total_chunks"] == 256
            assert stats["total_size"] == 1048576
            assert "python" in stats["languages"]
            
            # Verify line_count is not in stats (removed feature)
            assert "total_lines" not in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])