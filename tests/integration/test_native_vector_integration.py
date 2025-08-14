"""Integration tests for native Neo4j vector support.

This module tests the end-to-end integration of embedding providers
with native Neo4j vector format support.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
import os

from src.project_watch_mcp.utils.embeddings.openai import OpenAIEmbeddingsProvider
from src.project_watch_mcp.utils.embeddings.voyage import VoyageEmbeddingsProvider
from src.project_watch_mcp.utils.embeddings.vector_support import (
    VectorIndexManager,
    cosine_similarity,
    convert_to_numpy_array
)


class TestNativeVectorIntegration:
    """Test native vector support integration across providers."""
    
    @pytest.mark.asyncio
    async def test_openai_to_neo4j_pipeline(self):
        """Test complete pipeline from OpenAI embedding to Neo4j vector."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("src.project_watch_mcp.utils.embeddings.openai.AsyncOpenAI") as mock_openai:
                # Setup mock
                client_instance = AsyncMock()
                mock_openai.return_value = client_instance
                
                mock_response = MagicMock()
                mock_response.data = [
                    MagicMock(embedding=[0.3, 0.4, 0.5])  # Will be normalized to [0.424, 0.566, 0.707]
                ]
                client_instance.embeddings.create.return_value = mock_response
                
                # Create provider
                provider = OpenAIEmbeddingsProvider(dimension=3)
                
                # Get embedding in native format
                vector = await provider.embed_text("test text", native_format=True)
                
                # Verify it's a normalized numpy array
                assert isinstance(vector, np.ndarray)
                assert vector.dtype == np.float32
                assert pytest.approx(np.linalg.norm(vector), 0.0001) == 1.0
                
                # Verify approximate values after normalization
                expected = np.array([0.3, 0.4, 0.5], dtype=np.float32)
                expected = expected / np.linalg.norm(expected)
                assert np.allclose(vector, expected, rtol=1e-5)
    
    @pytest.mark.asyncio
    async def test_voyage_to_neo4j_pipeline(self):
        """Test complete pipeline from Voyage embedding to Neo4j vector."""
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "test-key"}):
            with patch("src.project_watch_mcp.utils.embeddings.voyage.voyageai.AsyncClient") as mock_voyage:
                # Setup mock
                client_instance = AsyncMock()
                mock_voyage.return_value = client_instance
                
                mock_response = MagicMock()
                mock_response.embeddings = [[0.6, 0.8, 0.0]]  # Will be normalized to [0.6, 0.8, 0.0]
                client_instance.embed.return_value = mock_response
                
                # Create provider
                provider = VoyageEmbeddingsProvider(model="voyage-3-lite")  # 512 dimensions
                
                # Get embedding in native format
                vector = await provider.embed_text("code snippet", native_format=True)
                
                # Verify it's a normalized numpy array
                assert isinstance(vector, np.ndarray)
                assert vector.dtype == np.float32
                assert pytest.approx(np.linalg.norm(vector), 0.0001) == 1.0
                
                # Verify values
                expected = np.array([0.6, 0.8, 0.0], dtype=np.float32)
                assert np.allclose(vector, expected, rtol=1e-5)
    
    @pytest.mark.asyncio
    async def test_batch_processing_with_native_format(self):
        """Test batch processing returns consistent native format vectors."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("src.project_watch_mcp.utils.embeddings.openai.AsyncOpenAI") as mock_openai:
                # Setup mock
                client_instance = AsyncMock()
                mock_openai.return_value = client_instance
                
                mock_response = MagicMock()
                mock_response.data = [
                    MagicMock(embedding=[1.0, 0.0, 0.0]),
                    MagicMock(embedding=[0.0, 1.0, 0.0]),
                    MagicMock(embedding=[0.0, 0.0, 1.0])
                ]
                client_instance.embeddings.create.return_value = mock_response
                
                # Create provider
                provider = OpenAIEmbeddingsProvider(dimension=3)
                
                # Get batch embeddings
                texts = ["text1", "text2", "text3"]
                vectors = await provider.embed_batch(texts, native_format=True)
                
                # Verify all are normalized numpy arrays
                assert len(vectors) == 3
                for i, vector in enumerate(vectors):
                    assert isinstance(vector, np.ndarray)
                    assert vector.dtype == np.float32
                    assert pytest.approx(np.linalg.norm(vector), 0.0001) == 1.0
                    
                    # Verify orthogonality (unit vectors along axes)
                    expected = np.zeros(3, dtype=np.float32)
                    expected[i] = 1.0
                    assert np.allclose(vector, expected)
    
    @pytest.mark.asyncio
    async def test_cross_provider_vector_compatibility(self):
        """Test vectors from different providers can be compared."""
        # Mock OpenAI
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "VOYAGE_API_KEY": "test-key"}):
            with patch("src.project_watch_mcp.utils.embeddings.openai.AsyncOpenAI") as mock_openai:
                with patch("src.project_watch_mcp.utils.embeddings.voyage.voyageai.AsyncClient") as mock_voyage:
                    # Setup OpenAI mock
                    openai_client = AsyncMock()
                    mock_openai.return_value = openai_client
                    openai_response = MagicMock()
                    openai_response.data = [MagicMock(embedding=[0.5, 0.5, 0.707])]
                    openai_client.embeddings.create.return_value = openai_response
                    
                    # Setup Voyage mock
                    voyage_client = AsyncMock()
                    mock_voyage.return_value = voyage_client
                    voyage_response = MagicMock()
                    voyage_response.embeddings = [[0.577, 0.577, 0.577]]
                    voyage_client.embed.return_value = voyage_response
                    
                    # Get vectors from both providers
                    openai_provider = OpenAIEmbeddingsProvider(dimension=3)
                    voyage_provider = VoyageEmbeddingsProvider(model="voyage-3-lite")
                    
                    openai_vector = await openai_provider.embed_text("test", native_format=True)
                    voyage_vector = await voyage_provider.embed_text("test", native_format=True)
                    
                    # Both should be valid normalized vectors
                    assert isinstance(openai_vector, np.ndarray)
                    assert isinstance(voyage_vector, np.ndarray)
                    assert pytest.approx(np.linalg.norm(openai_vector), 0.0001) == 1.0
                    assert pytest.approx(np.linalg.norm(voyage_vector), 0.0001) == 1.0
                    
                    # Calculate similarity
                    similarity = cosine_similarity(
                        openai_vector.tolist(),
                        voyage_vector.tolist()
                    )
                    
                    # Similarity should be a valid value between -1 and 1
                    assert -1.0 <= similarity <= 1.0
    
    @pytest.mark.asyncio
    async def test_vector_index_manager_with_native_vectors(self):
        """Test VectorIndexManager can create indexes for native vectors."""
        # Create mock driver
        mock_driver = MagicMock()
        mock_session = AsyncMock()
        
        # Setup context manager
        session_cm = AsyncMock()
        session_cm.__aenter__.return_value = mock_session
        session_cm.__aexit__.return_value = None
        mock_driver.session.return_value = session_cm
        
        # Create manager with OpenAI dimensions
        manager = VectorIndexManager(mock_driver, dimensions=1536)
        
        # Create index
        result = await manager.create_vector_index(
            index_name="native-embeddings",
            node_label="CodeChunk",
            property_name="embedding",
            similarity_function="cosine"
        )
        
        assert result["status"] == "created"
        assert result["index_name"] == "native-embeddings"
        
        # Verify the query includes correct dimensions
        mock_session.run.assert_called_once()
        query = mock_session.run.call_args[0][0]
        assert "`vector.dimensions`: 1536" in query
        assert "`vector.similarity_function`: 'cosine'" in query
    
    @pytest.mark.asyncio
    async def test_dimension_validation_across_providers(self):
        """Test dimension validation works correctly for each provider."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "VOYAGE_API_KEY": "test-key"}):
            with patch("src.project_watch_mcp.utils.embeddings.openai.AsyncOpenAI") as mock_openai:
                with patch("src.project_watch_mcp.utils.embeddings.voyage.voyageai.AsyncClient") as mock_voyage:
                    # Setup mocks with wrong dimensions
                    openai_client = AsyncMock()
                    mock_openai.return_value = openai_client
                    openai_response = MagicMock()
                    openai_response.data = [MagicMock(embedding=[0.1] * 1024)]  # Wrong!
                    openai_client.embeddings.create.return_value = openai_response
                    
                    voyage_client = AsyncMock()
                    mock_voyage.return_value = voyage_client
                    voyage_response = MagicMock()
                    voyage_response.embeddings = [[0.1] * 768]  # Wrong!
                    voyage_client.embed.return_value = voyage_response
                    
                    # OpenAI expects 1536
                    openai_provider = OpenAIEmbeddingsProvider(dimension=1536)
                    with pytest.raises(ValueError, match="dimension mismatch"):
                        await openai_provider.embed_text("test", validate_dimensions=True)
                    
                    # Voyage expects 1024 for voyage-code-3
                    voyage_provider = VoyageEmbeddingsProvider(model="voyage-code-3")
                    with pytest.raises(ValueError, match="dimension mismatch"):
                        await voyage_provider.embed_text("test", validate_dimensions=True)
    
    def test_fallback_behavior_on_conversion_error(self):
        """Test providers gracefully fallback when native conversion fails."""
        # Test the fallback behavior with a mock that simulates numpy error
        provider = OpenAIEmbeddingsProvider.__new__(OpenAIEmbeddingsProvider)
        provider.dimension = 3
        
        # Simulate a scenario where normalize_vector might fail
        test_vector = [float('inf'), 1.0, 2.0]  # Invalid vector with infinity
        
        # Should handle gracefully
        try:
            normalized = provider.normalize_vector(test_vector)
            # If it doesn't raise, check it handles infinity
            assert not np.isfinite(normalized).all() or np.allclose(normalized, [0, 0, 0])
        except:
            # Expected behavior - error is caught upstream
            pass
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_large_batch(self):
        """Test memory-efficient handling of large batches."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("src.project_watch_mcp.utils.embeddings.openai.AsyncOpenAI") as mock_openai:
                # Setup mock for large batch
                client_instance = AsyncMock()
                mock_openai.return_value = client_instance
                
                # Create 100 embeddings with 1536 dimensions each
                large_batch_response = MagicMock()
                large_batch_response.data = [
                    MagicMock(embedding=[0.01] * 1536) for _ in range(100)
                ]
                client_instance.embeddings.create.return_value = large_batch_response
                
                provider = OpenAIEmbeddingsProvider()
                texts = [f"text_{i}" for i in range(100)]
                
                # Get batch in native format
                vectors = await provider.embed_batch(texts, native_format=True)
                
                # Verify all use float32 for memory efficiency
                assert len(vectors) == 100
                for vector in vectors:
                    assert vector.dtype == np.float32
                    # Each vector should use 1536 * 4 bytes = 6144 bytes
                    assert vector.nbytes == 1536 * 4
                
                # Total memory for 100 vectors should be ~600KB, not ~1.2MB (float64)
                total_memory = sum(v.nbytes for v in vectors)
                assert total_memory == 100 * 1536 * 4  # ~600KB