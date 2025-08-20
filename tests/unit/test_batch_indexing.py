"""Tests for Neo4j batch indexing operations."""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from project_watch_mcp.neo4j_rag import CodeFile, Neo4jRAG


@pytest.fixture
def mock_neo4j_driver():
    """Create a mock Neo4j driver."""
    driver = AsyncMock()
    driver.execute_query = AsyncMock()
    return driver


@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings provider."""
    embeddings = AsyncMock()
    embeddings.embed_text = AsyncMock(return_value=[0.1] * 128)
    return embeddings


@pytest.fixture
async def neo4j_rag(mock_neo4j_driver, mock_embeddings):
    """Create a Neo4jRAG instance with mocked dependencies."""
    rag = Neo4jRAG(
        neo4j_driver=mock_neo4j_driver,
        project_name="test_project",
        chunk_size=100,
        chunk_overlap=20,
        embeddings=mock_embeddings,
    )
    return rag


def create_code_file(path: str, content: str = "test content", size: int = 100):
    """Helper to create a CodeFile instance."""
    return CodeFile(
        project_name="test_project",
        path=Path(path),
        content=content,
        language="python",
        size=size,
        last_modified=datetime.now(),
    )


class TestBatchIndexing:
    """Test batch indexing functionality."""

    async def test_batch_index_files_empty_list(self, neo4j_rag):
        """Test batch indexing with empty file list."""
        await neo4j_rag.batch_index_files([])
        
        # Should not call any queries
        neo4j_rag.neo4j_driver.execute_query.assert_not_called()

    async def test_batch_index_files_single_batch(self, neo4j_rag):
        """Test batch indexing with files fitting in single batch."""
        files = [
            create_code_file(f"file{i}.py", f"content {i}\n" * 10)
            for i in range(5)
        ]
        
        await neo4j_rag.batch_index_files(files, batch_size=10)
        
        # Should have called execute_query for:
        # 1. Batch upsert files
        # 2. Delete existing chunks
        # 3. Create new chunks
        assert neo4j_rag.neo4j_driver.execute_query.call_count >= 3
        
        # Check that UNWIND was used in the queries
        calls = neo4j_rag.neo4j_driver.execute_query.call_args_list
        
        # First call should be file upsert with UNWIND
        file_upsert_call = calls[0]
        assert "UNWIND" in file_upsert_call[0][0]
        assert "files" in file_upsert_call[0][1]
        assert len(file_upsert_call[0][1]["files"]) == 5
        
        # Second call should be chunk deletion with UNWIND
        chunk_delete_call = calls[1]
        assert "UNWIND" in chunk_delete_call[0][0]
        assert "paths" in chunk_delete_call[0][1]

    async def test_batch_index_files_multiple_batches(self, neo4j_rag):
        """Test batch indexing with files requiring multiple batches."""
        files = [
            create_code_file(f"file{i}.py", f"content {i}\n" * 10)
            for i in range(25)
        ]
        
        await neo4j_rag.batch_index_files(files, batch_size=10)
        
        # Should process in 3 batches (10, 10, 5)
        calls = neo4j_rag.neo4j_driver.execute_query.call_args_list
        
        # Count file upsert calls (contain "MERGE (f:CodeFile")
        file_upsert_calls = [
            call for call in calls 
            if "MERGE (f:CodeFile" in call[0][0]
        ]
        assert len(file_upsert_calls) == 3  # 3 batches
        
        # Check batch sizes
        assert len(file_upsert_calls[0][0][1]["files"]) == 10
        assert len(file_upsert_calls[1][0][1]["files"]) == 10
        assert len(file_upsert_calls[2][0][1]["files"]) == 5

    async def test_batch_index_files_with_large_content(self, neo4j_rag):
        """Test batch indexing with files having large content requiring chunking."""
        # Create files with content that will be chunked
        large_content = "\n".join([f"Line {i}" for i in range(500)])
        files = [
            create_code_file(f"file{i}.py", large_content, size=len(large_content))
            for i in range(3)
        ]
        
        await neo4j_rag.batch_index_files(files)
        
        calls = neo4j_rag.neo4j_driver.execute_query.call_args_list
        
        # Find chunk creation calls
        chunk_create_calls = [
            call for call in calls 
            if "CREATE (c:CodeChunk" in call[0][0]
        ]
        
        # Should have created chunks
        assert len(chunk_create_calls) > 0
        
        # Verify chunks have required fields
        for call in chunk_create_calls:
            chunks_data = call[0][1]["chunks"]
            assert len(chunks_data) > 0
            for chunk in chunks_data:
                assert "project_name" in chunk
                assert "file_path" in chunk
                assert "content" in chunk
                assert "start_line" in chunk
                assert "end_line" in chunk
                assert "chunk_index" in chunk

    async def test_batch_index_files_with_embeddings(self, neo4j_rag):
        """Test batch indexing generates embeddings when enabled."""
        files = [
            create_code_file(f"file{i}.py", f"content {i}\n" * 10)
            for i in range(3)
        ]
        
        await neo4j_rag.batch_index_files(files)
        
        # Embeddings should have been generated for each chunk
        assert neo4j_rag.embeddings.embed_text.called
        
        # Check that embeddings are included in chunk data
        calls = neo4j_rag.neo4j_driver.execute_query.call_args_list
        chunk_create_calls = [
            call for call in calls 
            if "CREATE (c:CodeChunk" in call[0][0]
        ]
        
        for call in chunk_create_calls:
            chunks_data = call[0][1]["chunks"]
            for chunk in chunks_data:
                assert "embedding" in chunk
                assert chunk["embedding"] is not None

    async def test_batch_index_files_handles_oversized_chunks(self, neo4j_rag, caplog):
        """Test batch indexing handles oversized chunks gracefully."""
        # Create content that would produce an oversized chunk
        # This is simulated since _sanitize_for_lucene should prevent this
        oversized_content = "x" * 35000  # Over 32KB when encoded
        files = [
            create_code_file("oversized.py", oversized_content)
        ]
        
        # Mock _sanitize_for_lucene to return oversized content for testing
        with patch.object(neo4j_rag, '_sanitize_for_lucene', return_value=oversized_content):
            await neo4j_rag.batch_index_files(files)
        
        # Should log error about oversized chunk
        assert "Skipping oversized chunk" in caplog.text

    async def test_batch_index_files_project_name_correction(self, neo4j_rag):
        """Test batch indexing corrects project names if needed."""
        # Create files with wrong project name
        files = [
            CodeFile(
                project_name="wrong_project",
                path=Path("file.py"),
                content="content",
                language="python",
                size=100,
                last_modified=datetime.now(),
            )
        ]
        
        await neo4j_rag.batch_index_files(files)
        
        # Check that files were indexed with correct project name
        calls = neo4j_rag.neo4j_driver.execute_query.call_args_list
        file_upsert_call = calls[0]
        
        file_data = file_upsert_call[0][1]["files"][0]
        assert file_data["project_name"] == "test_project"

    async def test_batch_index_performance_vs_single(self, neo4j_rag):
        """Test that batch indexing uses fewer database calls than single indexing."""
        files = [
            create_code_file(f"file{i}.py", f"content {i}\n" * 10)
            for i in range(10)
        ]
        
        # Batch indexing
        await neo4j_rag.batch_index_files(files)
        batch_call_count = neo4j_rag.neo4j_driver.execute_query.call_count
        
        # Reset mock
        neo4j_rag.neo4j_driver.execute_query.reset_mock()
        
        # Single file indexing (simulated)
        for file in files:
            await neo4j_rag.index_file(file)
        single_call_count = neo4j_rag.neo4j_driver.execute_query.call_count
        
        # Batch indexing should use significantly fewer calls
        # Each single index_file makes at least 3 calls per file (file node, delete chunks, create chunks)
        # Batch should make far fewer calls total
        assert batch_call_count < single_call_count / 2  # At least 2x improvement

    async def test_batch_index_files_chunk_sub_batching(self, neo4j_rag):
        """Test that large numbers of chunks are processed in sub-batches."""
        # Create files that will generate many chunks
        # With chunk_size=100 and overlap=20, we need much more content to exceed 1000 chunks
        large_content = "\n".join([f"Line {i}" for i in range(10000)])  # Will create ~125 chunks per file
        files = [
            create_code_file(f"file{i}.py", large_content, size=len(large_content))
            for i in range(10)  # 10 files * ~125 chunks = ~1250 chunks total
        ]
        
        await neo4j_rag.batch_index_files(files)
        
        # Find chunk creation calls
        calls = neo4j_rag.neo4j_driver.execute_query.call_args_list
        chunk_create_calls = [
            call for call in calls 
            if "CREATE (c:CodeChunk" in call[0][0]
        ]
        
        # Should have multiple chunk creation calls due to sub-batching (>1000 chunks)
        assert len(chunk_create_calls) >= 2
        
        # Each sub-batch should not exceed 1000 chunks
        total_chunks = 0
        for call in chunk_create_calls:
            chunks_data = call[0][1]["chunks"]
            assert len(chunks_data) <= 1000
            total_chunks += len(chunks_data)
        
        # Should have created a substantial number of chunks
        assert total_chunks > 1000