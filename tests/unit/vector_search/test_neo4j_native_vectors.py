"""Unit tests for Neo4j native vector index implementation.

Tests the native vector index creation, upsert, and search functions
following TDD principles.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any
import numpy as np

from project_watch_mcp.vector_search.neo4j_native_vectors import (
    NativeVectorIndex,
    VectorSearchResult,
    VectorUpsertResult,
    VectorIndexConfig,
)


class TestVectorIndexConfig:
    """Test vector index configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = VectorIndexConfig()
        
        assert config.index_name == "code-embeddings"
        assert config.node_label == "CodeChunk"
        assert config.embedding_property == "embedding"
        assert config.dimensions == 1536
        assert config.similarity_metric == "cosine"
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = VectorIndexConfig(
            index_name="custom-index",
            node_label="Document",
            embedding_property="vector",
            dimensions=3072,
            similarity_metric="euclidean"
        )
        
        assert config.index_name == "custom-index"
        assert config.node_label == "Document"
        assert config.embedding_property == "vector"
        assert config.dimensions == 3072
        assert config.similarity_metric == "euclidean"
        
    def test_voyage_dimensions(self):
        """Test Voyage AI embedding dimensions."""
        config = VectorIndexConfig(provider="voyage", dimensions=1024)
        assert config.dimensions == 1024
        
    def test_openai_large_dimensions(self):
        """Test OpenAI text-embedding-3-large dimensions."""
        config = VectorIndexConfig(provider="openai", dimensions=3072)
        assert config.dimensions == 3072


class TestNativeVectorIndex:
    """Test NativeVectorIndex class."""
    
    @pytest.fixture
    def mock_driver(self):
        """Create mock Neo4j driver."""
        driver = MagicMock()
        session = AsyncMock()
        driver.session.return_value.__aenter__.return_value = session
        driver.session.return_value.__aexit__.return_value = None
        return driver, session
    
    @pytest.fixture
    def vector_index(self, mock_driver):
        """Create NativeVectorIndex instance with mock driver."""
        driver, _ = mock_driver
        config = VectorIndexConfig()
        return NativeVectorIndex(driver, config)
    
    @pytest.mark.asyncio
    async def test_create_index(self, vector_index, mock_driver):
        """Test vector index creation."""
        _, session = mock_driver
        session.run.return_value = AsyncMock()
        
        result = await vector_index.create_index()
        
        assert result["status"] == "created"
        assert result["index_name"] == "code-embeddings"
        session.run.assert_called_once()
        
        # Verify the query contains correct parameters
        query = session.run.call_args[0][0]
        assert "CREATE VECTOR INDEX" in query
        assert "`code-embeddings`" in query
        assert "CodeChunk" in query
        assert "1536" in query
        assert "cosine" in query
    
    @pytest.mark.asyncio
    async def test_drop_index(self, vector_index, mock_driver):
        """Test dropping vector index."""
        _, session = mock_driver
        session.run.return_value = AsyncMock()
        
        result = await vector_index.drop_index()
        
        assert result["status"] == "dropped"
        assert result["index_name"] == "code-embeddings"
        session.run.assert_called_once_with(
            "DROP INDEX `code-embeddings` IF EXISTS"
        )
    
    @pytest.mark.asyncio
    async def test_upsert_single_vector(self, vector_index, mock_driver):
        """Test upserting a single vector."""
        _, session = mock_driver
        session.run.return_value = AsyncMock()
        
        vector = [0.1] * 1536
        metadata = {"file_path": "test.py", "line_number": 42}
        
        result = await vector_index.upsert_vector(
            node_id="test-node-1",
            vector=vector,
            metadata=metadata
        )
        
        assert isinstance(result, VectorUpsertResult)
        assert result.success
        assert result.node_id == "test-node-1"
        assert result.operation == "upserted"
        
        # Verify query structure
        query = session.run.call_args[0][0]
        assert "MERGE" in query
        assert "SET" in query
        assert "embedding" in query
    
    @pytest.mark.asyncio
    async def test_batch_upsert_vectors(self, vector_index, mock_driver):
        """Test batch upserting multiple vectors."""
        _, session = mock_driver
        
        # Mock the response to return node IDs
        mock_response = AsyncMock()
        mock_response.__aiter__.return_value = [
            {"node_id": f"node-{i}"} for i in range(5)
        ]
        session.run.return_value = mock_response
        
        vectors = [
            {
                "node_id": f"node-{i}",
                "vector": [0.1 * i] * 1536,
                "metadata": {"index": i}
            }
            for i in range(5)
        ]
        
        results = await vector_index.batch_upsert_vectors(vectors)
        
        assert len(results) == 5
        assert all(r.success for r in results)
        assert all(r.operation == "upserted" for r in results)
        
        # Verify batch query
        query = session.run.call_args[0][0]
        assert "UNWIND" in query
        assert "MERGE" in query
    
    @pytest.mark.asyncio
    async def test_search_vectors(self, vector_index, mock_driver):
        """Test vector similarity search."""
        _, session = mock_driver
        
        # Mock search results
        mock_results = [
            {
                "node": {"id": "node-1", "content": "test content 1"},
                "score": 0.95
            },
            {
                "node": {"id": "node-2", "content": "test content 2"},
                "score": 0.85
            }
        ]
        
        mock_response = AsyncMock()
        mock_response.__aiter__.return_value = [
            {"node": r["node"], "score": r["score"]} for r in mock_results
        ]
        session.run.return_value = mock_response
        
        query_vector = [0.2] * 1536
        results = await vector_index.search(
            vector=query_vector,
            top_k=10
        )
        
        assert len(results) == 2
        assert results[0].node_id == "node-1"
        assert results[0].score == 0.95
        assert results[1].node_id == "node-2"
        assert results[1].score == 0.85
        
        # Verify search query
        query = session.run.call_args[0][0]
        assert "CALL db.index.vector.queryNodes" in query
        assert "code-embeddings" in query
    
    @pytest.mark.asyncio
    async def test_search_with_metadata_filter(self, vector_index, mock_driver):
        """Test vector search with metadata filtering."""
        _, session = mock_driver
        
        mock_results = [
            {
                "node": {"id": "node-1", "language": "python"},
                "score": 0.9
            }
        ]
        
        mock_response = AsyncMock()
        mock_response.__aiter__.return_value = [
            {"node": r["node"], "score": r["score"]} for r in mock_results
        ]
        session.run.return_value = mock_response
        
        query_vector = [0.3] * 1536
        results = await vector_index.search(
            vector=query_vector,
            top_k=5,
            metadata_filter={"language": "python"}
        )
        
        assert len(results) == 1
        assert results[0].metadata["language"] == "python"
        
        # Verify filter in query
        query = session.run.call_args[0][0]
        assert "WHERE" in query
        assert "language" in query
    
    @pytest.mark.asyncio
    async def test_get_index_stats(self, vector_index, mock_driver):
        """Test getting index statistics."""
        _, session = mock_driver
        
        # First query returns index metadata
        mock_index_response = AsyncMock()
        mock_index_response.__aiter__.return_value = [{
            "name": "code-embeddings",
            "labelsOrTypes": ["CodeChunk"],
            "properties": ["embedding"],
            "options": {
                "indexConfig": {
                    "vector.dimensions": 1536,
                    "vector.similarity_function": "cosine"
                }
            }
        }]
        
        # Second query returns node count
        mock_count_response = AsyncMock()
        mock_count_response.single.return_value = {"node_count": 1000}
        
        # Set up side effects for the two queries
        session.run.side_effect = [mock_index_response, mock_count_response]
        
        stats = await vector_index.get_index_stats()
        
        assert stats["node_count"] == 1000
        assert stats["dimensions"] == 1536
        assert stats["similarity_function"] == "cosine"
    
    @pytest.mark.asyncio
    async def test_optimize_index(self, vector_index, mock_driver):
        """Test index optimization."""
        _, session = mock_driver
        session.run.return_value = AsyncMock()
        
        result = await vector_index.optimize_index()
        
        assert result["status"] == "optimized"
        assert "optimization_time_ms" in result
        
        # Verify optimization query
        query = session.run.call_args[0][0]
        assert "db.index.vector.optimize" in query or "ANALYZE" in query
    
    @pytest.mark.asyncio
    async def test_validate_vector_dimensions(self, vector_index):
        """Test vector dimension validation."""
        # Valid dimensions
        valid_vector = [0.1] * 1536
        assert vector_index._validate_dimensions(valid_vector) is True
        
        # Invalid dimensions
        invalid_vector = [0.1] * 100
        with pytest.raises(ValueError, match="Expected 1536"):
            vector_index._validate_dimensions(invalid_vector)
    
    @pytest.mark.asyncio
    async def test_normalize_vector(self, vector_index):
        """Test vector normalization."""
        vector = [3.0, 4.0]  # Simple 3-4-5 triangle
        normalized = vector_index._normalize_vector(vector)
        
        # Should have unit length
        assert pytest.approx(np.linalg.norm(normalized)) == 1.0
        assert pytest.approx(normalized[0]) == 0.6
        assert pytest.approx(normalized[1]) == 0.8


class TestVectorSearchResult:
    """Test VectorSearchResult dataclass."""
    
    def test_search_result_creation(self):
        """Test creating search result."""
        result = VectorSearchResult(
            node_id="test-node",
            score=0.95,
            metadata={"file": "test.py"},
            content="test content"
        )
        
        assert result.node_id == "test-node"
        assert result.score == 0.95
        assert result.metadata["file"] == "test.py"
        assert result.content == "test content"
    
    def test_search_result_comparison(self):
        """Test search results are sortable by score."""
        result1 = VectorSearchResult("node1", 0.8, {}, "")
        result2 = VectorSearchResult("node2", 0.9, {}, "")
        result3 = VectorSearchResult("node3", 0.7, {}, "")
        
        sorted_results = sorted([result1, result2, result3], 
                               key=lambda x: x.score, reverse=True)
        
        assert sorted_results[0].node_id == "node2"
        assert sorted_results[1].node_id == "node1"
        assert sorted_results[2].node_id == "node3"


class TestVectorUpsertResult:
    """Test VectorUpsertResult dataclass."""
    
    def test_upsert_result_success(self):
        """Test successful upsert result."""
        result = VectorUpsertResult(
            node_id="test-node",
            success=True,
            operation="created"
        )
        
        assert result.node_id == "test-node"
        assert result.success is True
        assert result.operation == "created"
        assert result.error is None
    
    def test_upsert_result_failure(self):
        """Test failed upsert result."""
        result = VectorUpsertResult(
            node_id="test-node",
            success=False,
            operation="failed",
            error="Dimension mismatch"
        )
        
        assert result.node_id == "test-node"
        assert result.success is False
        assert result.operation == "failed"
        assert result.error == "Dimension mismatch"


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.fixture
    def mock_driver(self):
        """Create mock Neo4j driver."""
        driver = MagicMock()
        session = AsyncMock()
        driver.session.return_value.__aenter__.return_value = session
        driver.session.return_value.__aexit__.return_value = None
        return driver, session
    
    @pytest.mark.asyncio
    async def test_full_index_lifecycle(self, mock_driver):
        """Test complete index lifecycle."""
        driver, session = mock_driver
        
        config = VectorIndexConfig(
            index_name="test-index",
            dimensions=3072,
            similarity_metric="cosine"
        )
        index = NativeVectorIndex(driver, config)
        
        # Create index
        session.run.return_value = AsyncMock()
        create_result = await index.create_index()
        assert create_result["status"] == "created"
        
        # Upsert vectors
        mock_response = AsyncMock()
        mock_response.__aiter__.return_value = [
            {"node_id": f"doc-{i}"} for i in range(10)
        ]
        session.run.return_value = mock_response
        
        vectors = [
            {"node_id": f"doc-{i}", "vector": [0.1 * i] * 3072, "metadata": {"doc_id": i}}
            for i in range(10)
        ]
        upsert_results = await index.batch_upsert_vectors(vectors)
        assert len(upsert_results) == 10
        
        # Search
        mock_response = AsyncMock()
        mock_response.__aiter__.return_value = [
            {"node": {"id": "doc-5"}, "score": 0.99}
        ]
        session.run.return_value = mock_response
        
        search_results = await index.search([0.5] * 3072, top_k=5)
        assert len(search_results) > 0
        
        # Get stats
        mock_index_response = AsyncMock()
        mock_index_response.__aiter__.return_value = [{
            "name": "test-index",
            "options": {
                "indexConfig": {
                    "vector.dimensions": 3072,
                    "vector.similarity_function": "cosine"
                }
            }
        }]
        mock_count_response = AsyncMock()
        mock_count_response.single.return_value = {"node_count": 10}
        # Reset side_effect properly
        session.run.side_effect = None
        session.run.side_effect = [mock_index_response, mock_count_response]
        
        stats = await index.get_index_stats()
        assert stats["node_count"] == 10
        
        # Drop index - reset to simple mock
        session.run.side_effect = None
        session.run.return_value = AsyncMock()
        drop_result = await index.drop_index()
        assert drop_result["status"] == "dropped"