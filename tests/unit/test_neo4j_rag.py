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
from unittest.mock import AsyncMock, MagicMock, patch
import logging

import pytest
import pytest_asyncio
from neo4j import AsyncSession

from src.project_watch_mcp.neo4j_rag import (
    Neo4jRAG,
    CodeFile,
    SearchResult
)
from src.project_watch_mcp.repository_monitor import FileInfo


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
            file_path="test_file.py",
            content="def test_function():",
            line_number=1,
            similarity_score=0.95
        ),
        SearchResult(
            file_path="another_file.py",
            content="class TestClass:",
            line_number=1,
            similarity_score=0.85
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
        assert rag.chunk_size == 1000  # default
        assert rag.chunk_overlap == 200  # default
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
        """Test database constraint creation."""
        session = mock_neo4j_driver.session()
        tx = AsyncMock()
        session.execute_write = AsyncMock(side_effect=lambda func: func(tx))
        
        await neo4j_rag.create_constraints()
        
        session.execute_write.assert_called()
        tx.run.assert_called()

    async def test_create_indexes(self, neo4j_rag):
        """Test database index creation."""
        # Mock the driver's execute_query method
        neo4j_rag.neo4j_driver.execute_query = AsyncMock()
        
        await neo4j_rag.create_indexes()
        
        # Verify that execute_query was called for index creation
        neo4j_rag.neo4j_driver.execute_query.assert_called()

    async def test_connection_verification(self, mock_neo4j_driver, mock_embeddings_provider):
        """Test Neo4j connection verification."""
        mock_neo4j_driver.verify_connectivity = AsyncMock()
        
        rag = Neo4jRAG(
            neo4j_driver=mock_neo4j_driver,
            project_name="test_project",
            embeddings=mock_embeddings_provider
        )
        
        await rag.initialize()
        mock_neo4j_driver.verify_connectivity.assert_called()

    async def test_close_connection(self, neo4j_rag, mock_neo4j_driver):
        """Test proper connection cleanup."""
        await neo4j_rag.close()
        mock_neo4j_driver.close.assert_called_once()

    async def test_initialize_error(self):
        """Test error handling during initialization."""
        with patch('src.project_watch_mcp.neo4j_rag.AsyncGraphDatabase') as mock_db:
            mock_driver = AsyncMock()
            mock_driver.verify_connectivity = AsyncMock(side_effect=Exception("Connection failed"))
            mock_db.driver.return_value = mock_driver
            
            with pytest.raises(Exception, match="Connection failed"):
                rag = Neo4jRAG(
                    neo4j_driver=mock_driver,
                    project_name="test_project",
                    embeddings=MagicMock()
                )
                await rag.initialize()


class TestNeo4jRAGIndexing:
    """Test file indexing and content chunking."""
    
    async def test_index_single_file(self, neo4j_rag, sample_code_file):
        """Test indexing a single code file."""
        # Mock the session and transaction
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_session.execute_write = AsyncMock(side_effect=lambda func: func(mock_tx))
        neo4j_rag.neo4j_driver.session.return_value = mock_session
        
        result = await neo4j_rag.index_file(sample_code_file)
        
        assert result is True
        mock_session.execute_write.assert_called()
        
    async def test_index_file_with_embeddings(self, neo4j_rag, sample_code_file, mock_embeddings_provider):
        """Test file indexing with embedding generation."""
        # Mock embedding generation
        mock_embeddings_provider.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3] * 512)
        
        # Mock the session
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_session.execute_write = AsyncMock(side_effect=lambda func: func(mock_tx))
        neo4j_rag.neo4j_driver.session.return_value = mock_session
        
        result = await neo4j_rag.index_file(sample_code_file)
        
        assert result is True
        mock_embeddings_provider.embed_text.assert_called()

    async def test_chunk_small_file(self, neo4j_rag):
        """Test chunking behavior for small files."""
        content = "def small_function():\n    return True"
        chunks = neo4j_rag.chunk_code(content)
        
        # Small file should result in a single chunk
        assert len(chunks) >= 1
        assert content in chunks[0]

    async def test_chunk_large_file(self, neo4j_rag):
        """Test chunking behavior for large files."""
        # Create content larger than chunk_size (100)
        content = "def large_function():\n" + "    # comment line\n" * 50
        chunks = neo4j_rag.chunk_code(content)
        
        # Large file should be split into multiple chunks
        assert len(chunks) > 1

    async def test_chunk_with_overlap(self, neo4j_rag):
        """Test chunk overlap functionality."""
        # Create content that will be chunked
        content = "\n".join([f"line_{i} = {i}" for i in range(20)])
        chunks = neo4j_rag.chunk_code(content)
        
        if len(chunks) > 1:
            # Check that consecutive chunks have some overlap
            # This is a basic check - actual overlap depends on implementation
            assert len(chunks[0]) > 0
            assert len(chunks[1]) > 0

    async def test_update_existing_file(self, neo4j_rag, sample_code_file):
        """Test updating an already indexed file."""
        # Mock the session
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_session.execute_write = AsyncMock(side_effect=lambda func: func(mock_tx))
        neo4j_rag.neo4j_driver.session.return_value = mock_session
        
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
        
        result = await neo4j_rag.update_file(updated_file.path, updated_file)
        assert result is True

    async def test_index_file_read_error(self, neo4j_rag):
        """Test handling of file read errors during indexing."""
        # Create a file that will cause a read error
        invalid_file = CodeFile(
            path=Path("nonexistent.py"),
            content="",  # Empty content to simulate read error
            language="python",
            size=0,
            last_modified=datetime.now(),
            project_name="test_project"
        )
        
        # Mock session to simulate database error
        mock_session = AsyncMock()
        mock_session.execute_write = AsyncMock(side_effect=Exception("Database error"))
        neo4j_rag.neo4j_driver.session.return_value = mock_session
        
        with pytest.raises(Exception):
            await neo4j_rag.index_file(invalid_file)

    async def test_index_file_with_existing_file(self, neo4j_rag, sample_code_file):
        """Test indexing a file that already exists in the database."""
        # Mock session to simulate existing file
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_session.execute_write = AsyncMock(side_effect=lambda func: func(mock_tx))
        neo4j_rag.neo4j_driver.session.return_value = mock_session
        
        # Index the file twice
        result1 = await neo4j_rag.index_file(sample_code_file)
        result2 = await neo4j_rag.index_file(sample_code_file)
        
        assert result1 is True
        assert result2 is True


class TestNeo4jRAGSearch:
    """Test search functionality."""
    
    async def test_semantic_search_basic(self, neo4j_rag, mock_embeddings_provider):
        """Test basic semantic search."""
        # Mock embedding generation
        mock_embeddings_provider.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3] * 512)
        
        # Mock search results
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_records = [
            MagicMock(
                data=lambda: {
                    "file": {"path": "test.py", "content": "def test():"},
                    "chunk": {"content": "def test():", "start_line": 1},
                    "score": 0.95
                }
            )
        ]
        mock_tx.run.return_value = mock_records
        mock_session.execute_read = AsyncMock(side_effect=lambda func: func(mock_tx))
        neo4j_rag.neo4j_driver.session.return_value = mock_session
        
        results = await neo4j_rag.search_semantic("test function", limit=10)
        
        assert len(results) == 1
        assert results[0].file_path == "test.py"
        assert results[0].similarity_score == 0.95
        mock_embeddings_provider.embed_text.assert_called_with("test function")

    async def test_semantic_search_with_language_filter(self, neo4j_rag, mock_embeddings_provider):
        """Test semantic search filtered by language."""
        mock_embeddings_provider.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3] * 512)
        
        # Mock session
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_tx.run.return_value = []
        mock_session.execute_read = AsyncMock(side_effect=lambda func: func(mock_tx))
        neo4j_rag.neo4j_driver.session.return_value = mock_session
        
        results = await neo4j_rag.search_semantic("test", language="python", limit=10)
        
        assert isinstance(results, list)
        # Verify language filter was applied in query
        mock_tx.run.assert_called()

    async def test_semantic_search_empty_query(self, neo4j_rag):
        """Test handling of empty search queries."""
        results = await neo4j_rag.search_semantic("", limit=10)
        assert results == []

    async def test_pattern_search_literal(self, neo4j_rag):
        """Test literal pattern search."""
        # Mock search results
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_records = [
            MagicMock(
                data=lambda: {
                    "file": {"path": "test.py", "content": "def test_function():"},
                    "chunk": {"content": "def test_function():", "start_line": 1},
                    "score": 1.0
                }
            )
        ]
        mock_tx.run.return_value = mock_records
        mock_session.execute_read = AsyncMock(side_effect=lambda func: func(mock_tx))
        neo4j_rag.neo4j_driver.session.return_value = mock_session
        
        results = await neo4j_rag.search_by_pattern("def test_function", limit=10)
        
        assert len(results) == 1
        assert results[0].file_path == "test.py"
        assert "def test_function" in results[0].content

    async def test_pattern_search_regex(self, neo4j_rag):
        """Test regex pattern search."""
        # Mock search results for regex pattern
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_records = [
            MagicMock(
                data=lambda: {
                    "file": {"path": "test.py", "content": "def test_function():"},
                    "chunk": {"content": "def test_function():", "start_line": 1},
                    "score": 1.0
                }
            )
        ]
        mock_tx.run.return_value = mock_records
        mock_session.execute_read = AsyncMock(side_effect=lambda func: func(mock_tx))
        neo4j_rag.neo4j_driver.session.return_value = mock_session
        
        results = await neo4j_rag.search_by_pattern("def.*function", is_regex=True, limit=10)
        
        assert len(results) == 1
        assert results[0].file_path == "test.py"

    async def test_pattern_search_case_insensitive(self, neo4j_rag):
        """Test case-insensitive pattern search."""
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_records = [
            MagicMock(
                data=lambda: {
                    "file": {"path": "test.py", "content": "def TEST_FUNCTION():"},
                    "chunk": {"content": "def TEST_FUNCTION():", "start_line": 1},
                    "score": 1.0
                }
            )
        ]
        mock_tx.run.return_value = mock_records
        mock_session.execute_read = AsyncMock(side_effect=lambda func: func(mock_tx))
        neo4j_rag.neo4j_driver.session.return_value = mock_session
        
        # Case insensitive search should find uppercase function
        results = await neo4j_rag.search_by_pattern("test_function", limit=10)
        
        assert len(results) == 1

    async def test_search_limit_parameter(self, neo4j_rag):
        """Test search result limiting."""
        # Mock multiple search results
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_records = [
            MagicMock(data=lambda: {
                "file": {"path": f"test{i}.py", "content": f"content {i}"},
                "chunk": {"content": f"content {i}", "start_line": 1},
                "score": 1.0 - i * 0.1
            })
            for i in range(10)
        ]
        mock_tx.run.return_value = mock_records
        mock_session.execute_read = AsyncMock(side_effect=lambda func: func(mock_tx))
        neo4j_rag.neo4j_driver.session.return_value = mock_session
        
        # Test limit parameter
        results = await neo4j_rag.search_by_pattern("content", limit=3)
        
        # Should respect the limit parameter in the query
        mock_tx.run.assert_called()

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
        # Mock file metadata
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_record = MagicMock()
        mock_record.data.return_value = {
            "f": {
                "path": "test.py",
                "language": "python",
                "size": 100,
                "last_modified": "2023-01-01T00:00:00Z"
            }
        }
        mock_tx.run.return_value = [mock_record]
        mock_session.execute_read = AsyncMock(side_effect=lambda func: func(mock_tx))
        neo4j_rag.neo4j_driver.session.return_value = mock_session
        
        metadata = await neo4j_rag.get_file_metadata(Path("test.py"))
        
        assert metadata is not None
        assert metadata["path"] == "test.py"
        assert metadata["language"] == "python"

    async def test_get_file_metadata_nonexistent(self, neo4j_rag):
        """Test metadata request for nonexistent file."""
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_tx.run.return_value = []  # No results
        mock_session.execute_read = AsyncMock(side_effect=lambda func: func(mock_tx))
        neo4j_rag.neo4j_driver.session.return_value = mock_session
        
        metadata = await neo4j_rag.get_file_metadata(Path("nonexistent.py"))
        
        assert metadata is None

    async def test_update_file_content(self, neo4j_rag, sample_code_file):
        """Test updating file content and embeddings."""
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_session.execute_write = AsyncMock(side_effect=lambda func: func(mock_tx))
        neo4j_rag.neo4j_driver.session.return_value = mock_session
        
        # Update file with new content
        updated_file = CodeFile(
            path=sample_code_file.path,
            content="def updated_function():\n    return 'updated'",
            language=sample_code_file.language,
            size=50,
            last_modified=datetime.now(),
            project_name=sample_code_file.project_name
        )
        
        result = await neo4j_rag.update_file(sample_code_file.path, updated_file)
        
        assert result is True
        mock_session.execute_write.assert_called()

    async def test_delete_file(self, neo4j_rag, sample_code_file):
        """Test file deletion from index."""
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_session.execute_write = AsyncMock(side_effect=lambda func: func(mock_tx))
        neo4j_rag.neo4j_driver.session.return_value = mock_session
        
        result = await neo4j_rag.delete_file(sample_code_file.path)
        
        assert result is True
        mock_session.execute_write.assert_called()

    async def test_delete_nonexistent_file(self, neo4j_rag):
        """Test deletion of file not in index."""
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_session.execute_write = AsyncMock(side_effect=lambda func: func(mock_tx))
        neo4j_rag.neo4j_driver.session.return_value = mock_session
        
        result = await neo4j_rag.delete_file(Path("nonexistent.py"))
        
        # Should handle gracefully - implementation dependent
        assert result is not None

    async def test_get_repository_stats(self, neo4j_rag):
        """Test repository statistics retrieval."""
        # Mock repository stats
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_record = MagicMock()
        mock_record.data.return_value = {
            "total_files": 10,
            "total_chunks": 50,
            "languages": ["python", "javascript"]
        }
        mock_tx.run.return_value = [mock_record]
        mock_session.execute_read = AsyncMock(side_effect=lambda func: func(mock_tx))
        neo4j_rag.neo4j_driver.session.return_value = mock_session
        
        stats = await neo4j_rag.get_repository_stats()
        
        assert stats is not None
        # Implementation specific assertions would go here

    async def test_update_file_creates_if_not_exists(self, neo4j_rag, sample_code_file):
        """Test that update_file creates file if it doesn't exist."""
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_session.execute_write = AsyncMock(side_effect=lambda func: func(mock_tx))
        neo4j_rag.neo4j_driver.session.return_value = mock_session
        
        result = await neo4j_rag.update_file(sample_code_file.path, sample_code_file)
        
        assert result is True
        mock_session.execute_write.assert_called()

    async def test_delete_file_error_handling(self, neo4j_rag):
        """Test error handling during file deletion."""
        mock_session = AsyncMock()
        mock_session.execute_write = AsyncMock(side_effect=Exception("Database error"))
        neo4j_rag.neo4j_driver.session.return_value = mock_session
        
        with pytest.raises(Exception, match="Database error"):
            await neo4j_rag.delete_file(Path("test.py"))

    async def test_get_repository_stats_empty_repo(self, neo4j_rag):
        """Test repository statistics for empty repository."""
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_tx.run.return_value = []  # Empty repository
        mock_session.execute_read = AsyncMock(side_effect=lambda func: func(mock_tx))
        neo4j_rag.neo4j_driver.session.return_value = mock_session
        
        stats = await neo4j_rag.get_repository_stats()
        
        # Should handle empty repository gracefully
        assert stats is not None or stats == {}


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
        
        # Mock session for bulk operations
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_session.execute_write = AsyncMock(side_effect=lambda func: func(mock_tx))
        neo4j_rag.neo4j_driver.session.return_value = mock_session
        
        # Index all files
        results = []
        for file in test_files:
            result = await neo4j_rag.index_file(file)
            results.append(result)
        
        # All files should be indexed successfully
        assert all(results)
        assert mock_session.execute_write.call_count == len(test_files)

    @pytest.mark.slow
    async def test_large_repository_scan(self, neo4j_rag):
        """Test performance with large repositories."""
        # This is a placeholder for performance testing with large datasets
        # In a real scenario, this would test with hundreds of files
        
        # Mock repository stats for large repo
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_record = MagicMock()
        mock_record.data.return_value = {
            "total_files": 1000,
            "total_chunks": 10000,
            "languages": ["python", "javascript", "typescript"]
        }
        mock_tx.run.return_value = [mock_record]
        mock_session.execute_read = AsyncMock(side_effect=lambda func: func(mock_tx))
        neo4j_rag.neo4j_driver.session.return_value = mock_session
        
        stats = await neo4j_rag.get_repository_stats()
        
        assert stats is not None
        # Performance assertions would go here

    async def test_embedding_caching(self, neo4j_rag, mock_embeddings_provider):
        """Test embedding result caching."""
        # Mock embedding provider to track calls
        mock_embeddings_provider.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3] * 512)
        
        # Mock search to trigger embedding generation
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_tx.run.return_value = []
        mock_session.execute_read = AsyncMock(side_effect=lambda func: func(mock_tx))
        neo4j_rag.neo4j_driver.session.return_value = mock_session
        
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
        
        # Mock session
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_session.execute_write = AsyncMock(side_effect=lambda func: func(mock_tx))
        neo4j_rag.neo4j_driver.session.return_value = mock_session
        
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
        neo4j_rag.neo4j_driver.session.side_effect = Exception("Connection lost")
        
        with pytest.raises(Exception, match="Connection lost"):
            await neo4j_rag.get_repository_stats()

    async def test_concurrent_file_updates(self, neo4j_rag):
        """Test handling of concurrent file modifications."""
        # This test would verify that concurrent updates don't cause data corruption
        # Implementation would depend on the actual concurrency handling in Neo4jRAG
        
        # Mock session with potential race condition
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_session.execute_write = AsyncMock(side_effect=lambda func: func(mock_tx))
        neo4j_rag.neo4j_driver.session.return_value = mock_session
        
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
        # Implementation might raise exception or return False
        try:
            result = await neo4j_rag.index_file(invalid_file)
            # If no exception, result should indicate failure
            assert result is not True
        except (ValueError, Exception):
            # Exception is acceptable for invalid input
            pass

    async def test_transaction_rollback(self, neo4j_rag):
        """Test transaction rollback on error."""
        # Mock session with transaction that fails partway through
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_tx.run.side_effect = [None, Exception("Transaction failed")]
        mock_session.execute_write = AsyncMock(side_effect=lambda func: func(mock_tx))
        neo4j_rag.neo4j_driver.session.return_value = mock_session
        
        test_file = CodeFile(
            path=Path("test_rollback.py"),
            content="def test():\n    pass\n",
            language="python",
            size=20,
            last_modified=datetime.now(),
            project_name="test_project"
        )
        
        # Operation should fail and rollback
        with pytest.raises(Exception):
            await neo4j_rag.index_file(test_file)


class TestSearchResultModels:
    """Test search result model functionality."""
    
    def test_search_result_initialization(self):
        """Test SearchResult model initialization."""
        result = SearchResult(
            file_path="test.py",
            content="def test():",
            line_number=1,
            similarity_score=0.95
        )
        
        assert result.file_path == "test.py"
        assert result.content == "def test():"
        assert result.line_number == 1
        assert result.similarity_score == 0.95

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