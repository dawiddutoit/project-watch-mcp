"""Extended tests for Neo4j RAG module to improve coverage."""

import hashlib
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from neo4j.exceptions import Neo4jError

from project_watch_mcp.neo4j_rag import CodeFile, Neo4jRAG, SearchResult


class TestNeo4jRAGExtended:
    """Extended tests for Neo4j RAG functionality."""

    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        driver = AsyncMock()
        driver.verify_connectivity = AsyncMock()
        driver.close = AsyncMock()
        
        # Mock execute_query to return empty records by default
        driver.execute_query = AsyncMock()
        driver.execute_query.return_value = AsyncMock(records=[])
        return driver

    @pytest.fixture
    def mock_embeddings_provider(self):
        """Create a mock embeddings provider."""
        provider = AsyncMock()
        provider.embed_text.return_value = [0.1] * 1536
        provider.embed_batch.return_value = [[0.1] * 1536]
        provider.dimension = 1536
        return provider

    @pytest.fixture
    def neo4j_rag(self, mock_driver, mock_embeddings_provider):
        """Create a Neo4j RAG instance with mocked dependencies."""
        rag = Neo4jRAG(
            neo4j_driver=mock_driver,
            project_name="test_project",
            embeddings=mock_embeddings_provider
        )
        return rag

    @pytest.mark.asyncio
    async def test_close(self, neo4j_rag, mock_driver):
        """Test closing the Neo4j connection."""
        await neo4j_rag.close()
        mock_driver.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_indexes_error_handling(self, neo4j_rag, mock_driver):
        """Test error handling in create_indexes."""
        mock_driver.execute_query.side_effect = Neo4jError("Index creation failed")

        # Should log error but not raise
        await neo4j_rag.create_indexes()

    @pytest.mark.asyncio
    async def test_chunk_file_content_small_file(self, neo4j_rag):
        """Test chunking small files."""
        content = "def hello():\n    print('world')"
        chunks = neo4j_rag.chunk_content(content, chunk_size=100, overlap=10)

        assert len(chunks) == 1
        assert chunks[0] == content

    @pytest.mark.asyncio
    async def test_chunk_file_content_large_file(self, neo4j_rag):
        """Test chunking large files with overlap."""
        lines = [f"line {i}" for i in range(20)]
        content = "\n".join(lines)

        chunks = neo4j_rag.chunk_content(content, chunk_size=50, overlap=10)

        assert len(chunks) > 1
        # Check that chunks overlap
        for i in range(len(chunks) - 1):
            # The last 'overlap' characters of chunk i should be in chunk i+1
            overlap_text = chunks[i][-10:] if len(chunks[i]) >= 10 else chunks[i]
            assert overlap_text in chunks[i + 1]

    @pytest.mark.asyncio
    async def test_index_file_with_existing_file(self, neo4j_rag, mock_driver):
        """Test indexing when file already exists."""
        # Mock successful execution
        mock_driver.execute_query.return_value = AsyncMock(records=[])

        code_file = CodeFile(
            project_name="test_project",
            path=Path("/test/file.py"),
            content="print('test')",
            language="python",
            size=100,
            last_modified=datetime.now(),
        )

        await neo4j_rag.index_file(code_file)

        # Should call execute_query multiple times (for file and chunks)
        calls = mock_driver.execute_query.call_args_list
        assert any("DELETE" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_update_file_creates_if_not_exists(self, neo4j_rag, mock_driver):
        """Test update_file creates file if it doesn't exist."""
        # Mock successful execution
        mock_driver.execute_query.return_value = AsyncMock(records=[])

        code_file = CodeFile(
            project_name="test_project",
            path=Path("/test/new.py"),
            content="new content",
            language="python",
            size=100,
            last_modified=datetime.now(),
        )

        await neo4j_rag.update_file(code_file)

        # Should create the file
        calls = mock_driver.execute_query.call_args_list
        assert any("MERGE" in str(call) or "CREATE" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_delete_file_error_handling(self, neo4j_rag, mock_driver):
        """Test error handling in delete_file."""
        mock_driver.execute_query.side_effect = Neo4jError("Delete failed")

        # Should not raise
        await neo4j_rag.delete_file(Path("/test/file.py"))

    @pytest.mark.asyncio
    async def test_search_semantic_empty_query(self, neo4j_rag):
        """Test semantic search with empty query."""
        results = await neo4j_rag.search_semantic("")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_semantic_with_language_filter(self, neo4j_rag, mock_driver, mock_embeddings_provider):
        """Test semantic search with language filter."""
        # Mock execute_query for semantic search
        mock_record = MagicMock()
        mock_record.__getitem__ = lambda self, key: {
            "file_path": "/test/file.py",
            "content": "test content",
            "chunk_id": "chunk1",
            "start_line": 1,
            "end_line": 10,
            "score": 0.95,
            "project_name": "test_project",
        }[key]
        mock_record.get = lambda key, default=None: {
            "file_path": "/test/file.py",
            "content": "test content",
            "chunk_id": "chunk1",
            "start_line": 1,
            "end_line": 10,
            "score": 0.95,
            "project_name": "test_project",
        }.get(key, default)
        
        mock_driver.execute_query.return_value = AsyncMock(records=[mock_record])

        results = await neo4j_rag.search_semantic("test", language="python")

        assert len(results) == 1
        assert results[0].file_path == "/test/file.py"

    @pytest.mark.asyncio
    async def test_search_by_pattern_with_regex(self, neo4j_rag, mock_driver):
        """Test pattern search with regex."""
        # Mock execute_query for pattern search
        mock_record = MagicMock()
        mock_record.__getitem__ = lambda self, key: {
            "file_path": "/test/file.py",
            "content": "def test_function():",
            "chunk_id": "chunk1",
            "start_line": 5,
            "end_line": 5,
            "project_name": "test_project",
        }[key]
        mock_record.get = lambda key, default=None: {
            "file_path": "/test/file.py",
            "content": "def test_function():",
            "chunk_id": "chunk1",
            "start_line": 5,
            "end_line": 5,
            "project_name": "test_project",
        }.get(key, default)
        
        mock_driver.execute_query.return_value = AsyncMock(records=[mock_record])

        results = await neo4j_rag.search_by_pattern(r"def \w+\(\):", is_regex=True)

        assert len(results) == 1
        assert results[0].content == "def test_function():"

    @pytest.mark.asyncio
    async def test_get_file_metadata_with_existing_file(self, neo4j_rag, mock_driver):
        """Test getting metadata for existing file."""
        # Mock execute_query for metadata query
        mock_record = MagicMock()
        mock_record.__getitem__ = lambda self, key: {
            "path": "/test/file.py",
            "language": "python",
            "size": 1000,
            "last_modified": "2024-01-01T10:00:00",
            "hash": "abc123",
            "chunk_count": 5,
            "project_name": "test_project",
        }[key]
        mock_record.get = lambda key, default=None: {
            "path": "/test/file.py",
            "language": "python",
            "size": 1000,
            "last_modified": "2024-01-01T10:00:00",
            "hash": "abc123",
            "chunk_count": 5,
            "project_name": "test_project",
        }.get(key, default)
        
        mock_driver.execute_query.return_value = AsyncMock(records=[mock_record])

        metadata = await neo4j_rag.get_file_metadata(Path("/test/file.py"))

        assert metadata["language"] == "python"
        assert metadata["size"] == 1000
        assert metadata["chunk_count"] == 5

    @pytest.mark.asyncio
    async def test_get_file_metadata_nonexistent(self, neo4j_rag, mock_driver):
        """Test getting metadata for non-existent file."""
        # Mock execute_query to return no records
        mock_driver.execute_query.return_value = AsyncMock(records=[])

        metadata = await neo4j_rag.get_file_metadata(Path("/test/missing.py"))

        assert metadata is None

    @pytest.mark.asyncio
    async def test_get_repository_stats_empty_repo(self, neo4j_rag, mock_driver):
        """Test getting stats for empty repository."""
        # Mock execute_query for stats query
        mock_record = MagicMock()
        mock_record.__getitem__ = lambda self, key: {
            "total_files": 0,
            "total_chunks": 0,
            "total_size": 0,
            "languages": [],
            "project_name": "test_project",
        }[key]
        mock_record.get = lambda key, default=None: {
            "total_files": 0,
            "total_chunks": 0,
            "total_size": 0,
            "languages": [],
            "project_name": "test_project",
        }.get(key, default)
        
        mock_driver.execute_query.return_value = AsyncMock(records=[mock_record])

        stats = await neo4j_rag.get_repository_stats()

        assert stats["total_files"] == 0
        assert stats["total_chunks"] == 0
        assert stats["languages"] == []

    @pytest.mark.asyncio
    async def test_search_result_initialization(self):
        """Test SearchResult dataclass."""
        result = SearchResult(
            project_name="test_project",
            file_path="/test/file.py",
            content="test content",
            line_number=10,
            similarity=0.85,
        )

        assert result.file_path == "/test/file.py"
        assert result.similarity == 0.85

    @pytest.mark.asyncio
    async def test_code_file_hash_property(self):
        """Test CodeFile hash property."""
        code_file = CodeFile(
            project_name="test_project",
            path=Path("/test/file.py"),
            content="test content",
            language="python",
            size=100,
            last_modified=datetime.now(),
        )

        expected_hash = hashlib.sha256(b"test content").hexdigest()
        assert code_file.file_hash == expected_hash

    @pytest.mark.asyncio
    async def test_initialize_error(self, mock_driver, mock_embeddings_provider):
        """Test handling Neo4j initialization errors."""
        rag = Neo4jRAG(
            neo4j_driver=mock_driver,
            project_name="test_project",
            embeddings=mock_embeddings_provider
        )

        # Mock execute_query to raise error during initialization
        mock_driver.execute_query.side_effect = Exception("Initialization failed")

        # Should not raise but log error
        await rag.initialize()
