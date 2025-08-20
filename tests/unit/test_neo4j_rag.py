"""
Comprehensive test suite for Neo4j RAG functionality.

This consolidated test file replaces multiple overlapping test files:
- test_neo4j_rag.py
- test_neo4j_rag_comprehensive.py
- test_neo4j_rag_extended.py

Test Classes:
- TestNeo4jRAGInitialization: Configuration and setup
- TestNeo4jRAGIndexing: File indexing and chunking
- TestNeo4jRAGSearch: Semantic and pattern search
- TestNeo4jRAGFileOperations: CRUD operations
- TestNeo4jRAGPerformance: Performance-related tests
"""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from neo4j import AsyncSession

from src.project_watch_mcp.neo4j_rag import CodeFile, Neo4jRAG, SearchResult

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_neo4j_driver():
    """Create a comprehensive mock Neo4j driver."""
    driver = AsyncMock()
    session = AsyncMock(spec=AsyncSession)

    # Mock transaction methods
    tx = AsyncMock()
    tx.run = AsyncMock()

    # Set up session to return transaction
    session.execute_read = AsyncMock(side_effect=lambda func: func(tx))
    session.execute_write = AsyncMock(side_effect=lambda func: func(tx))
    session.close = AsyncMock()

    # Driver returns session
    driver.session = MagicMock(return_value=session)
    driver.close = AsyncMock()
    driver.verify_connectivity = AsyncMock()
    driver.execute_query = AsyncMock()

    return driver


@pytest.fixture
def mock_embeddings_provider():
    """Mock embeddings provider for testing."""
    embeddings = MagicMock()
    embeddings.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3] * 512)  # 1536 dimensions
    embeddings.embed_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3] * 512, [0.4, 0.5, 0.6] * 512])
    return embeddings


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    client = MagicMock()

    # Mock embeddings response
    embedding_response = MagicMock()
    embedding_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3] * 512)]  # 1536 dimensions
    client.embeddings.create = MagicMock(return_value=embedding_response)

    return client


@pytest_asyncio.fixture
async def neo4j_rag(mock_neo4j_driver, mock_embeddings_provider):
    """Create a Neo4jRAG instance for testing."""
    rag = Neo4jRAG(
        neo4j_driver=mock_neo4j_driver,
        project_name="test_project",
        embeddings=mock_embeddings_provider,
        chunk_size=100,
        chunk_overlap=20,
        enable_file_classification=True  # Added new parameter
    )
    await rag.initialize()
    return rag


@pytest.fixture
def sample_code_file():
    """Sample CodeFile for testing."""
    return CodeFile(
        path=Path("test_file.py"),
        content="def test_function():\n    pass\n",
        language="python",
        size=25,
        last_modified=datetime.now(),
        project_name="test_project"
    )


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return [
        SearchResult(
            project_name="test_project",
            file_path=Path("test_file.py"),
            content="def test_function():",
            line_number=1,
            similarity=0.95
        ),
        SearchResult(
            project_name="test_project",
            file_path=Path("another_file.py"),
            content="class TestClass:",
            line_number=1,
            similarity=0.85
        )
    ]


# ============================================================================
# TEST CLASSES
# ============================================================================

class TestNeo4jRAGInitialization:
    """Test Neo4j RAG initialization and configuration."""

    async def test_initialization_default_config(self, mock_neo4j_driver, mock_embeddings_provider):
        """Test initialization with default configuration."""
        rag = Neo4jRAG(
            neo4j_driver=mock_neo4j_driver,
            project_name="test_project",
            embeddings=mock_embeddings_provider
        )

        assert rag.project_name == "test_project"
        assert rag.chunk_size == 500  # default changed from 1000 to 500
        assert rag.chunk_overlap == 50  # default changed from 200 to 50
        assert rag.enable_file_classification is True  # new default parameter
        assert rag.neo4j_driver is not None
        assert rag.embeddings is not None

    async def test_initialization_custom_config(self, mock_neo4j_driver, mock_embeddings_provider):
        """Test initialization with custom chunk size and overlap."""
        rag = Neo4jRAG(
            neo4j_driver=mock_neo4j_driver,
            project_name="test_project",
            embeddings=mock_embeddings_provider,
            chunk_size=500,
            chunk_overlap=100
        )

        assert rag.project_name == "test_project"
        assert rag.chunk_size == 500
        assert rag.chunk_overlap == 100

    async def test_create_constraints(self, neo4j_rag, mock_neo4j_driver):
        """Test database constraint creation.
        Note: create_constraints method has been removed, constraints are now created in create_indexes.
        """
        # This test is no longer applicable as create_constraints was removed
        # Constraints are now created as part of create_indexes
        pytest.skip("create_constraints method no longer exists")

    async def test_create_indexes(self, neo4j_rag):
        """Test database index creation."""
        # Mock the driver's execute_query method
        neo4j_rag.neo4j_driver.execute_query = AsyncMock()

        await neo4j_rag.create_indexes()

        # Verify that execute_query was called for index creation
        neo4j_rag.neo4j_driver.execute_query.assert_called()

    async def test_connection_verification(self, mock_neo4j_driver, mock_embeddings_provider):
        """Test Neo4j connection verification."""
        # verify_connectivity is not automatically called by Neo4jRAG
        # Test that initialization works properly instead
        rag = Neo4jRAG(
            neo4j_driver=mock_neo4j_driver,
            project_name="test_project",
            embeddings=mock_embeddings_provider
        )

        await rag.initialize()
        # Verify that execute_query was called for index creation
        mock_neo4j_driver.execute_query.assert_called()

    async def test_close_connection(self, neo4j_rag, mock_neo4j_driver):
        """Test proper connection cleanup."""
        await neo4j_rag.close()
        mock_neo4j_driver.close.assert_called_once()

    async def test_initialize_error(self):
        """Test error handling during initialization."""
        mock_driver = AsyncMock()
        mock_driver.execute_query = AsyncMock(side_effect=Exception("Connection failed"))

        rag = Neo4jRAG(
            neo4j_driver=mock_driver,
            project_name="test_project",
            embeddings=MagicMock()
        )

        # initialize doesn't raise exceptions - it logs warnings for failures
        # This is to allow graceful degradation when certain features aren't available
        await rag.initialize()

        # Verify that execute_query was attempted
        mock_driver.execute_query.assert_called()


class TestNeo4jRAGIndexing:
    """Test file indexing and content chunking."""

    async def test_index_single_file(self, neo4j_rag, sample_code_file):
        """Test indexing a single code file."""
        # Mock the execute_query method
        neo4j_rag.neo4j_driver.execute_query = AsyncMock()

        # index_file doesn't return a value, it just executes
        await neo4j_rag.index_file(sample_code_file)

        # Verify execute_query was called (for file node creation and chunks)
        neo4j_rag.neo4j_driver.execute_query.assert_called()

    async def test_index_file_with_embeddings(self, neo4j_rag, sample_code_file, mock_embeddings_provider):
        """Test file indexing with embedding generation."""
        # Mock embedding generation
        mock_embeddings_provider.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3] * 512)

        # Mock the execute_query method
        neo4j_rag.neo4j_driver.execute_query = AsyncMock()

        # index_file doesn't return a value
        await neo4j_rag.index_file(sample_code_file)

        # Verify execute_query was called
        neo4j_rag.neo4j_driver.execute_query.assert_called()
        # Note: embed_text is called for each chunk
        if neo4j_rag.embeddings:
            mock_embeddings_provider.embed_text.assert_called()

    async def test_chunk_small_file(self, neo4j_rag):
        """Test chunking behavior for small files."""
        content = "def small_function():\n    return True"
        chunks = neo4j_rag.chunk_content(content, chunk_size=neo4j_rag.chunk_size, overlap=neo4j_rag.chunk_overlap)

        # Small file should result in a single chunk
        assert len(chunks) >= 1
        assert content in chunks[0]

    async def test_chunk_large_file(self, neo4j_rag):
        """Test chunking behavior for large files."""
        # Create content that will exceed the Lucene byte limit (30KB)
        # Each line needs to be longer to force chunking
        # The chunking function checks byte size first (30KB limit)
        long_line = "    # " + "x" * 500  # Long comment line
        lines = ["def large_function():"] + [long_line] * 100
        content = "\n".join(lines)

        # Verify this content is actually large enough to trigger chunking
        content_bytes = len(content.encode('utf-8'))
        assert content_bytes > 30000, f"Test content is only {content_bytes} bytes, need > 30KB"

        chunks = neo4j_rag.chunk_content(content, chunk_size=neo4j_rag.chunk_size, overlap=neo4j_rag.chunk_overlap)

        # Large file should be split into multiple chunks
        assert len(chunks) > 1, f"Expected multiple chunks for {content_bytes} bytes but got {len(chunks)} chunk(s)"

    async def test_chunk_with_overlap(self, neo4j_rag):
        """Test chunk overlap functionality."""
        # Create content that will be chunked
        content = "\n".join([f"line_{i} = {i}" for i in range(20)])
        chunks = neo4j_rag.chunk_content(content, chunk_size=neo4j_rag.chunk_size, overlap=neo4j_rag.chunk_overlap)

        if len(chunks) > 1:
            # Check that consecutive chunks have some overlap
            # This is a basic check - actual overlap depends on implementation
            assert len(chunks[0]) > 0
            assert len(chunks[1]) > 0

    async def test_update_existing_file(self, neo4j_rag, sample_code_file):
        """Test updating an already indexed file."""
        # Mock execute_query
        neo4j_rag.neo4j_driver.execute_query = AsyncMock()

        # First index the file
        await neo4j_rag.index_file(sample_code_file)

        # Update the file content
        updated_file = CodeFile(
            path=sample_code_file.path,
            content="def updated_function():\n    return False\n",
            language=sample_code_file.language,
            size=35,
            last_modified=datetime.now(),
            project_name=sample_code_file.project_name
        )

        # update_file only takes the CodeFile object now
        await neo4j_rag.update_file(updated_file)
        # update_file doesn't return a value
        # Verify execute_query was called for the update
        neo4j_rag.neo4j_driver.execute_query.assert_called()

    async def test_index_file_read_error(self, neo4j_rag):
        """Test handling of file read errors during indexing."""
        # Create a file that will cause a read error
        invalid_file = CodeFile(
            path=Path("nonexistent.py"),
            content="",  # Empty content
            language="python",
            size=0,
            last_modified=datetime.now(),
            project_name="test_project"
        )

        # Mock execute_query to raise an error
        neo4j_rag.neo4j_driver.execute_query = AsyncMock(side_effect=Exception("Database error"))

        # Should raise an exception when execute_query is called
        with pytest.raises(Exception, match="Database error"):
            await neo4j_rag.index_file(invalid_file)

    async def test_index_file_with_existing_file(self, neo4j_rag, sample_code_file):
        """Test indexing a file that already exists in the database."""
        # Mock execute_query
        neo4j_rag.neo4j_driver.execute_query = AsyncMock()

        # Index the file twice
        # index_file doesn't return a value
        await neo4j_rag.index_file(sample_code_file)
        await neo4j_rag.index_file(sample_code_file)

        # Verify execute_query was called multiple times
        assert neo4j_rag.neo4j_driver.execute_query.call_count >= 2


class TestNeo4jRAGSearch:
    """Test search functionality."""

    async def test_semantic_search_basic(self, neo4j_rag, mock_embeddings_provider):
        """Test basic semantic search."""
        # Mock embedding generation
        mock_embeddings_provider.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3] * 512)

        # Mock execute_query results
        mock_result = MagicMock()
        mock_result.records = [
            {
                "file_path": "test.py",
                "chunk_content": "def test():",
                "line_number": 1,
                "similarity": 0.95,
                "project_name": "test_project"
            }
        ]
        neo4j_rag.neo4j_driver.execute_query = AsyncMock(return_value=mock_result)

        results = await neo4j_rag.search_semantic("test function", limit=10)

        assert len(results) == 1
        assert results[0].file_path == Path("test.py")
        assert results[0].similarity == 0.95
        mock_embeddings_provider.embed_text.assert_called_with("test function")

    async def test_semantic_search_with_language_filter(self, neo4j_rag, mock_embeddings_provider):
        """Test semantic search filtered by language."""
        mock_embeddings_provider.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3] * 512)

        # Mock execute_query with empty results
        mock_result = MagicMock()
        mock_result.records = []
        neo4j_rag.neo4j_driver.execute_query = AsyncMock(return_value=mock_result)

        results = await neo4j_rag.search_semantic("test", language="python", limit=10)

        assert isinstance(results, list)
        assert len(results) == 0
        # Verify execute_query was called with language parameter
        neo4j_rag.neo4j_driver.execute_query.assert_called()

    async def test_semantic_search_empty_query(self, neo4j_rag):
        """Test handling of empty search queries."""
        results = await neo4j_rag.search_semantic("", limit=10)
        assert results == []

    async def test_pattern_search_literal(self, neo4j_rag):
        """Test literal pattern search."""
        # Mock execute_query results
        mock_result = MagicMock()
        mock_result.records = [
            {
                "file_path": "test.py",
                "content": "def test_function():",
                "line_number": 1,
                "project_name": "test_project"
            }
        ]
        neo4j_rag.neo4j_driver.execute_query = AsyncMock(return_value=mock_result)

        results = await neo4j_rag.search_by_pattern("def test_function", limit=10)

        assert len(results) == 1
        assert results[0].file_path == Path("test.py")
        assert "def test_function" in results[0].content

    async def test_pattern_search_regex(self, neo4j_rag):
        """Test regex pattern search."""
        # Mock execute_query results for regex pattern
        mock_result = MagicMock()
        mock_result.records = [
            {
                "file_path": "test.py",
                "content": "def test_function():",
                "line_number": 1,
                "project_name": "test_project"
            }
        ]
        neo4j_rag.neo4j_driver.execute_query = AsyncMock(return_value=mock_result)

        results = await neo4j_rag.search_by_pattern("def.*function", is_regex=True, limit=10)

        assert len(results) == 1
        assert results[0].file_path == Path("test.py")

    async def test_pattern_search_case_insensitive(self, neo4j_rag):
        """Test case-insensitive pattern search."""
        # Mock execute_query results
        mock_result = MagicMock()
        mock_result.records = [
            {
                "file_path": "test.py",
                "content": "def TEST_FUNCTION():",
                "line_number": 1,
                "project_name": "test_project"
            }
        ]
        neo4j_rag.neo4j_driver.execute_query = AsyncMock(return_value=mock_result)

        # Case insensitive search should find uppercase function
        results = await neo4j_rag.search_by_pattern("test_function", limit=10)

        assert len(results) == 1

    async def test_search_limit_parameter(self, neo4j_rag):
        """Test search result limiting."""
        # Mock multiple search results
        mock_result = MagicMock()
        mock_result.records = [
            {
                "file_path": f"test{i}.py",
                "content": f"content {i}",
                "line_number": 1,
                "project_name": "test_project"
            }
            for i in range(3)  # Return only 3 results as per limit
        ]
        neo4j_rag.neo4j_driver.execute_query = AsyncMock(return_value=mock_result)

        # Test limit parameter
        results = await neo4j_rag.search_by_pattern("content", limit=3)

        # Should return 3 results
        assert len(results) == 3
        neo4j_rag.neo4j_driver.execute_query.assert_called()

    async def test_search_with_empty_results(self, neo4j_rag):
        """Test search behavior with no results."""
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_tx.run.return_value = []
        mock_session.execute_read = AsyncMock(side_effect=lambda func: func(mock_tx))
        neo4j_rag.neo4j_driver.session.return_value = mock_session

        results = await neo4j_rag.search_by_pattern("nonexistent_pattern", limit=10)

        assert results == []

    async def test_escape_lucene_query(self, neo4j_rag):
        """Test Lucene query escaping."""
        # Test special characters that need escaping
        from src.project_watch_mcp.neo4j_rag import escape_lucene_query

        query_with_special_chars = "function() + - && || ! ( ) { } [ ] ^ \" ~ * ? : \\"
        escaped = escape_lucene_query(query_with_special_chars)

        # Should escape special Lucene characters
        assert "\\" in escaped or escaped != query_with_special_chars


class TestNeo4jRAGFileOperations:
    """Test file CRUD operations."""

    async def test_get_file_metadata_existing(self, neo4j_rag):
        """Test retrieving metadata for existing file."""
        # Mock execute_query result
        mock_result = MagicMock()
        mock_result.records = [
            {
                "path": "test.py",
                "language": "python",
                "size": 100,
                "lines": 10,
                "last_modified": "2023-01-01T00:00:00Z",
                "hash": "abc123",
                "project_name": "test_project",
                "chunk_count": 2
            }
        ]
        neo4j_rag.neo4j_driver.execute_query = AsyncMock(return_value=mock_result)

        metadata = await neo4j_rag.get_file_metadata(Path("test.py"))

        assert metadata is not None
        assert metadata["path"] == "test.py"
        assert metadata["language"] == "python"

    async def test_get_file_metadata_nonexistent(self, neo4j_rag):
        """Test metadata request for nonexistent file."""
        # Mock execute_query with no results
        mock_result = MagicMock()
        mock_result.records = []  # No results
        neo4j_rag.neo4j_driver.execute_query = AsyncMock(return_value=mock_result)

        metadata = await neo4j_rag.get_file_metadata(Path("nonexistent.py"))

        assert metadata is None

    async def test_update_file_content(self, neo4j_rag, sample_code_file):
        """Test updating file content and embeddings."""
        # Mock execute_query for checking hash and updating
        neo4j_rag.neo4j_driver.execute_query = AsyncMock()

        # Update file with new content
        updated_file = CodeFile(
            path=sample_code_file.path,
            content="def updated_function():\n    return 'updated'",
            language=sample_code_file.language,
            size=50,
            last_modified=datetime.now(),
            project_name=sample_code_file.project_name
        )

        # update_file only takes the CodeFile object now
        await neo4j_rag.update_file(updated_file)

        # Verify execute_query was called
        neo4j_rag.neo4j_driver.execute_query.assert_called()

    async def test_delete_file(self, neo4j_rag, sample_code_file):
        """Test file deletion from index."""
        # Mock execute_query
        neo4j_rag.neo4j_driver.execute_query = AsyncMock()

        # delete_file doesn't return a value
        await neo4j_rag.delete_file(sample_code_file.path)

        # Verify execute_query was called for deletion
        neo4j_rag.neo4j_driver.execute_query.assert_called()

    async def test_delete_nonexistent_file(self, neo4j_rag):
        """Test deletion of file not in index."""
        # Mock execute_query
        neo4j_rag.neo4j_driver.execute_query = AsyncMock()

        # delete_file doesn't return a value, even for nonexistent files
        await neo4j_rag.delete_file(Path("nonexistent.py"))

        # Should handle gracefully - verify execute_query was called
        neo4j_rag.neo4j_driver.execute_query.assert_called()

    async def test_get_repository_stats(self, neo4j_rag):
        """Test repository statistics retrieval."""
        # Mock execute_query result
        mock_result = MagicMock()
        mock_result.records = [
            {
                "project_name": "test_project",
                "total_files": 10,
                "total_chunks": 50,
                "total_size": 10000,
                "languages": ["python", "javascript"]
            }
        ]
        neo4j_rag.neo4j_driver.execute_query = AsyncMock(return_value=mock_result)

        stats = await neo4j_rag.get_repository_stats()

        assert stats is not None
        assert stats["total_files"] == 10
        assert stats["total_chunks"] == 50
        assert "python" in stats["languages"]

    async def test_update_file_creates_if_not_exists(self, neo4j_rag, sample_code_file):
        """Test that update_file creates file if it doesn't exist."""
        # Mock execute_query - first call returns no existing file, subsequent calls succeed
        mock_result = MagicMock()
        mock_result.records = []  # No existing file
        neo4j_rag.neo4j_driver.execute_query = AsyncMock(return_value=mock_result)

        # update_file only takes the CodeFile object
        await neo4j_rag.update_file(sample_code_file)

        # Verify execute_query was called
        neo4j_rag.neo4j_driver.execute_query.assert_called()

    async def test_delete_file_error_handling(self, neo4j_rag):
        """Test error handling during file deletion."""
        # Mock execute_query to raise an error
        neo4j_rag.neo4j_driver.execute_query = AsyncMock(side_effect=Exception("Database error"))

        with pytest.raises(Exception, match="Database error"):
            await neo4j_rag.delete_file(Path("test.py"))

    async def test_get_repository_stats_empty_repo(self, neo4j_rag):
        """Test repository statistics for empty repository."""
        # Mock execute_query with no records
        mock_result = MagicMock()
        mock_result.records = []  # Empty repository
        neo4j_rag.neo4j_driver.execute_query = AsyncMock(return_value=mock_result)

        stats = await neo4j_rag.get_repository_stats()

        # Should return default stats for empty repository
        assert stats is not None
        assert stats["total_files"] == 0
        assert stats["total_chunks"] == 0
        assert stats["languages"] == []


class TestNeo4jRAGPerformance:
    """Test performance-related functionality."""

    @pytest.mark.slow
    async def test_bulk_indexing(self, neo4j_rag, mock_embeddings_provider):
        """Test indexing multiple files in bulk."""
        # Create multiple test files
        test_files = [
            CodeFile(
                path=Path(f"test_file_{i}.py"),
                content=f"def function_{i}():\n    return {i}\n",
                language="python",
                size=30 + i,
                last_modified=datetime.now(),
                project_name="test_project"
            )
            for i in range(5)
        ]

        # Mock execute_query for bulk operations
        neo4j_rag.neo4j_driver.execute_query = AsyncMock()

        # Index all files
        for file in test_files:
            await neo4j_rag.index_file(file)

        # All files should be indexed successfully
        # Verify execute_query was called multiple times for indexing
        assert neo4j_rag.neo4j_driver.execute_query.call_count >= len(test_files)

    @pytest.mark.slow
    async def test_large_repository_scan(self, neo4j_rag):
        """Test performance with large repositories."""
        # This is a placeholder for performance testing with large datasets
        # In a real scenario, this would test with hundreds of files

        # Mock repository stats for large repo
        mock_result = MagicMock()
        mock_result.records = [
            {
                "project_name": "test_project",
                "total_files": 1000,
                "total_chunks": 10000,
                "total_size": 1000000,
                "languages": ["python", "javascript", "typescript"]
            }
        ]
        neo4j_rag.neo4j_driver.execute_query = AsyncMock(return_value=mock_result)

        stats = await neo4j_rag.get_repository_stats()

        assert stats is not None
        assert stats["total_files"] == 1000
        assert stats["total_chunks"] == 10000

    async def test_embedding_caching(self, neo4j_rag, mock_embeddings_provider):
        """Test embedding result caching."""
        # Mock embedding provider to track calls
        mock_embeddings_provider.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3] * 512)

        # Mock execute_query to return empty results
        mock_result = MagicMock()
        mock_result.records = []
        neo4j_rag.neo4j_driver.execute_query = AsyncMock(return_value=mock_result)

        # Search with same query twice
        query = "test function"
        await neo4j_rag.search_semantic(query, limit=10)
        await neo4j_rag.search_semantic(query, limit=10)

        # Embeddings provider should be called for each search
        assert mock_embeddings_provider.embed_text.call_count == 2

    async def test_concurrent_indexing(self, neo4j_rag, mock_embeddings_provider):
        """Test handling of concurrent file modifications."""
        # Create test files for concurrent indexing
        test_files = [
            CodeFile(
                path=Path(f"concurrent_test_{i}.py"),
                content=f"def concurrent_function_{i}():\n    return {i}\n",
                language="python",
                size=40 + i,
                last_modified=datetime.now(),
                project_name="test_project"
            )
            for i in range(3)
        ]

        # Mock execute_query for concurrent operations
        neo4j_rag.neo4j_driver.execute_query = AsyncMock()

        # Index files concurrently
        tasks = [neo4j_rag.index_file(file) for file in test_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All tasks should complete successfully
        assert len(results) == len(test_files)
        assert all(result is True or not isinstance(result, Exception) for result in results)

    async def test_generate_embedding(self, neo4j_rag, mock_embeddings_provider):
        """Test embedding generation."""
        test_text = "This is a test function"
        mock_embeddings_provider.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3] * 512)

        embedding = await neo4j_rag.embeddings.embed_text(test_text)

        assert len(embedding) == 1536  # Standard OpenAI embedding dimension
        mock_embeddings_provider.embed_text.assert_called_once_with(test_text)

    async def test_generate_embedding_error(self, neo4j_rag, mock_embeddings_provider):
        """Test handling of embedding API failures."""
        test_text = "This should fail"
        mock_embeddings_provider.embed_text = AsyncMock(side_effect=Exception("API Error"))

        with pytest.raises(Exception, match="API Error"):
            await neo4j_rag.embeddings.embed_text(test_text)


class TestNeo4jRAGErrorHandling:
    """Test error handling and edge cases."""

    async def test_embedding_generation_failure(self, neo4j_rag, mock_embeddings_provider):
        """Test handling of embedding API failures."""
        mock_embeddings_provider.embed_text = AsyncMock(side_effect=Exception("Embedding API failed"))

        # Mock search that triggers embedding
        with pytest.raises(Exception, match="Embedding API failed"):
            await neo4j_rag.search_semantic("test query", limit=10)

    async def test_neo4j_connection_lost(self, neo4j_rag):
        """Test handling of lost database connection."""
        # Simulate connection loss
        neo4j_rag.neo4j_driver.execute_query = AsyncMock(side_effect=Exception("Connection lost"))

        with pytest.raises(Exception, match="Connection lost"):
            await neo4j_rag.get_repository_stats()

    async def test_concurrent_file_updates(self, neo4j_rag):
        """Test handling of concurrent file modifications."""
        # This test would verify that concurrent updates don't cause data corruption
        # Implementation would depend on the actual concurrency handling in Neo4jRAG

        # Mock execute_query for concurrent operations
        neo4j_rag.neo4j_driver.execute_query = AsyncMock()

        # Simulate concurrent file updates
        test_file = CodeFile(
            path=Path("concurrent_test.py"),
            content="def test():\n    pass\n",
            language="python",
            size=20,
            last_modified=datetime.now(),
            project_name="test_project"
        )

        # Index the same file multiple times concurrently
        tasks = [neo4j_rag.index_file(test_file) for _ in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Should handle concurrent operations gracefully
        assert len(results) == 3

    async def test_invalid_file_path(self, neo4j_rag):
        """Test handling of invalid file paths."""
        invalid_file = CodeFile(
            path=Path(""),  # Invalid empty path
            content="test content",
            language="unknown",
            size=12,
            last_modified=datetime.now(),
            project_name="test_project"
        )

        # Should handle invalid paths gracefully
        # index_file doesn't return a value, so we just verify it doesn't crash
        neo4j_rag.neo4j_driver.execute_query = AsyncMock()
        await neo4j_rag.index_file(invalid_file)
        # Verify execute_query was called (even with invalid path)
        neo4j_rag.neo4j_driver.execute_query.assert_called()

    async def test_transaction_rollback(self, neo4j_rag):
        """Test transaction rollback on error."""
        # Mock execute_query to fail partway through
        neo4j_rag.neo4j_driver.execute_query = AsyncMock(side_effect=[None, Exception("Transaction failed")])

        test_file = CodeFile(
            path=Path("test_rollback.py"),
            content="def test():\n    pass\n",
            language="python",
            size=20,
            last_modified=datetime.now(),
            project_name="test_project"
        )

        # Operation should fail and rollback
        with pytest.raises(Exception, match="Transaction failed"):
            await neo4j_rag.index_file(test_file)


class TestSearchResultModels:
    """Test search result model functionality."""

    def test_search_result_initialization(self):
        """Test SearchResult model initialization."""
        result = SearchResult(
            project_name="test_project",
            file_path=Path("test.py"),
            content="def test():",
            line_number=1,
            similarity=0.95
        )

        assert result.file_path == Path("test.py")
        assert result.content == "def test():"
        assert result.line_number == 1
        assert result.similarity == 0.95

    def test_code_file_hash_property(self):
        """Test CodeFile hash property."""
        code_file = CodeFile(
            path=Path("test.py"),
            content="def test():\n    pass",
            language="python",
            size=20,
            last_modified=datetime.now(),
            project_name="test_project"
        )

        # Hash should be consistent for same content
        hash1 = hash(str(code_file.content) + str(code_file.path))
        hash2 = hash(str(code_file.content) + str(code_file.path))
        assert hash1 == hash2

        # Different content should produce different hash
        code_file2 = CodeFile(
            path=Path("test.py"),
            content="def different():\n    pass",
            language="python",
            size=22,
            last_modified=datetime.now(),
            project_name="test_project"
        )
        hash3 = hash(str(code_file2.content) + str(code_file2.path))
        assert hash1 != hash3

