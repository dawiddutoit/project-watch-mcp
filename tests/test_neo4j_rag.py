"""Test suite for Neo4j RAG functionality."""

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from src.project-watch.neo4j_rag import (
    CodeFile,
    Neo4jRAG,
)


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver for testing."""
    driver = AsyncMock()
    driver.verify_connectivity = AsyncMock()
    driver.execute_query = AsyncMock()
    return driver


@pytest.fixture
def mock_embeddings():
    """Mock embeddings generator."""
    embeddings = MagicMock()
    embeddings.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])
    embeddings.embed_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    return embeddings


@pytest_asyncio.fixture
async def neo4j_rag(mock_neo4j_driver, mock_embeddings):
    """Create a Neo4jRAG instance for testing."""
    rag = Neo4jRAG(
        neo4j_driver=mock_neo4j_driver,
        embeddings=mock_embeddings,
        chunk_size=100,
        chunk_overlap=20,
    )
    await rag.initialize()
    return rag


class TestNeo4jRAG:
    """Test suite for Neo4jRAG class."""

    async def test_initialization(self, neo4j_rag):
        """Test that Neo4jRAG initializes correctly."""
        assert neo4j_rag.chunk_size == 100
        assert neo4j_rag.chunk_overlap == 20
        assert neo4j_rag.neo4j_driver is not None
        assert neo4j_rag.embeddings is not None

    async def test_create_indexes(self, neo4j_rag):
        """Test creation of Neo4j indexes."""
        await neo4j_rag.create_indexes()

        # Verify that index creation queries were executed
        neo4j_rag.neo4j_driver.execute_query.assert_called()

    async def test_index_file(self, neo4j_rag):
        """Test indexing a single file."""
        code_file = CodeFile(
            path=Path("/test/file.py"),
            content="def hello():\n    print('Hello, World!')",
            language="python",
            size=35,
            last_modified=datetime.now(),
        )

        await neo4j_rag.index_file(code_file)

        # Verify that the file was indexed
        neo4j_rag.neo4j_driver.execute_query.assert_called()
        neo4j_rag.embeddings.embed_text.assert_called()

    async def test_chunk_file_content(self, neo4j_rag):
        """Test chunking of file content."""
        content = "Line 1\n" * 50  # Create content that needs chunking
        chunks = neo4j_rag.chunk_content(content, chunk_size=50, overlap=10)

        assert len(chunks) > 1
        # Verify overlap exists
        for i in range(len(chunks) - 1):
            assert chunks[i][-10:] in chunks[i + 1]

    async def test_search_semantic(self, neo4j_rag):
        """Test semantic search functionality."""
        # Mock search results
        neo4j_rag.neo4j_driver.execute_query.return_value = MagicMock(
            records=[
                {
                    "file_path": "/test/file.py",
                    "chunk_content": "def hello():",
                    "similarity": 0.95,
                    "line_number": 1,
                }
            ]
        )

        results = await neo4j_rag.search_semantic("hello function", limit=5)

        assert len(results) == 1
        assert results[0].file_path == Path("/test/file.py")
        assert results[0].similarity == 0.95
        assert "def hello():" in results[0].content

    async def test_search_by_pattern(self, neo4j_rag):
        """Test pattern-based search."""
        # Mock search results
        neo4j_rag.neo4j_driver.execute_query.return_value = MagicMock(
            records=[
                {
                    "file_path": "/test/file.py",
                    "content": "class TestClass:",
                    "line_number": 10,
                }
            ]
        )

        results = await neo4j_rag.search_by_pattern("class.*:", is_regex=True)

        assert len(results) == 1
        assert results[0].file_path == Path("/test/file.py")
        assert "class TestClass:" in results[0].content

    async def test_update_file(self, neo4j_rag):
        """Test updating an existing file in the graph."""
        code_file = CodeFile(
            path=Path("/test/file.py"),
            content="# Updated content",
            language="python",
            size=17,
            last_modified=datetime.now(),
        )

        await neo4j_rag.update_file(code_file)

        # Verify update queries were executed
        neo4j_rag.neo4j_driver.execute_query.assert_called()

    async def test_delete_file(self, neo4j_rag):
        """Test deleting a file from the graph."""
        file_path = Path("/test/file.py")

        await neo4j_rag.delete_file(file_path)

        # Verify deletion query was executed
        neo4j_rag.neo4j_driver.execute_query.assert_called()
        call_args = neo4j_rag.neo4j_driver.execute_query.call_args
        assert str(file_path) in str(call_args)

    async def test_get_file_metadata(self, neo4j_rag):
        """Test retrieving file metadata."""
        # Mock metadata result
        neo4j_rag.neo4j_driver.execute_query.return_value = MagicMock(
            records=[
                {
                    "path": "/test/file.py",
                    "language": "python",
                    "size": 100,
                    "last_modified": datetime.now().isoformat(),
                    "chunk_count": 3,
                    "hash": "abc123def456789",
                }
            ]
        )

        metadata = await neo4j_rag.get_file_metadata(Path("/test/file.py"))

        assert metadata is not None
        assert metadata["language"] == "python"
        assert metadata["size"] == 100
        assert metadata["chunk_count"] == 3

    async def test_get_repository_stats(self, neo4j_rag):
        """Test getting repository statistics."""
        # Mock stats result
        neo4j_rag.neo4j_driver.execute_query.return_value = MagicMock(
            records=[
                {
                    "total_files": 10,
                    "total_chunks": 50,
                    "total_size": 10000,
                    "languages": ["python", "javascript", "markdown"],
                }
            ]
        )

        stats = await neo4j_rag.get_repository_stats()

        assert stats["total_files"] == 10
        assert stats["total_chunks"] == 50
        assert stats["total_size"] == 10000
        assert len(stats["languages"]) == 3
