"""Unit tests for native Neo4j vector support in embedding providers.

This test module validates the TASK-005 requirements for updating the embedding
pipeline to support native Neo4j vector format with proper normalization and
batch processing capabilities.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from typing import List

from src.project_watch_mcp.utils.embeddings.base import EmbeddingsProvider
from src.project_watch_mcp.utils.embeddings.openai import OpenAIEmbeddingsProvider
from src.project_watch_mcp.utils.embeddings.voyage import VoyageEmbeddingsProvider


class TestNativeVectorSupport:
    """Test native Neo4j vector format support in base class."""
    
    def test_base_provider_has_native_vector_methods(self):
        """Test that base provider defines native vector conversion methods."""
        provider = EmbeddingsProvider()
        
        # Check for new methods
        assert hasattr(provider, "to_neo4j_vector")
        assert hasattr(provider, "normalize_vector")
        assert hasattr(provider, "embed_batch")
        assert hasattr(provider, "validate_vector_dimensions")
    
    def test_to_neo4j_vector_converts_list_to_numpy(self):
        """Test conversion of embedding list to Neo4j-compatible numpy array."""
        provider = EmbeddingsProvider()
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        neo4j_vector = provider.to_neo4j_vector(embedding)
        
        assert isinstance(neo4j_vector, np.ndarray)
        assert neo4j_vector.dtype == np.float32
        assert neo4j_vector.shape == (5,)
        assert np.allclose(neo4j_vector, np.array(embedding, dtype=np.float32))
    
    def test_normalize_vector_for_cosine_similarity(self):
        """Test vector normalization for cosine similarity in Neo4j."""
        provider = EmbeddingsProvider()
        vector = [3.0, 4.0, 0.0]  # Vector with magnitude 5
        
        normalized = provider.normalize_vector(vector)
        
        # Check it's normalized (magnitude = 1)
        magnitude = np.linalg.norm(normalized)
        assert pytest.approx(magnitude, 0.0001) == 1.0
        
        # Check proportions are maintained
        expected = np.array([0.6, 0.8, 0.0], dtype=np.float32)
        assert np.allclose(normalized, expected)
    
    def test_normalize_zero_vector_handling(self):
        """Test that zero vectors are handled gracefully."""
        provider = EmbeddingsProvider()
        zero_vector = [0.0, 0.0, 0.0]
        
        normalized = provider.normalize_vector(zero_vector)
        
        # Zero vector should remain zero after normalization
        assert np.allclose(normalized, np.zeros(3, dtype=np.float32))
    
    def test_validate_vector_dimensions(self):
        """Test dimension validation for vectors."""
        provider = EmbeddingsProvider(dimension=1536)
        
        # Valid dimension
        valid_vector = [0.1] * 1536
        assert provider.validate_vector_dimensions(valid_vector) is True
        
        # Invalid dimension
        invalid_vector = [0.1] * 1024
        assert provider.validate_vector_dimensions(invalid_vector) is False
    
    @pytest.mark.asyncio
    async def test_embed_batch_base_implementation(self):
        """Test batch embedding with native vector support."""
        provider = EmbeddingsProvider()
        texts = ["text1", "text2", "text3"]
        
        # Should raise NotImplementedError in base class
        with pytest.raises(NotImplementedError):
            await provider.embed_batch(texts)


class TestOpenAINativeVectorSupport:
    """Test OpenAI provider with native Neo4j vector support."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for testing."""
        with patch("src.project_watch_mcp.utils.embeddings.openai.AsyncOpenAI") as mock:
            client_instance = AsyncMock()
            mock.return_value = client_instance
            
            # Mock embedding response
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[0.1, 0.2, 0.3, 0.4, 0.5])
            ]
            client_instance.embeddings.create.return_value = mock_response
            
            yield client_instance
    
    @pytest.mark.asyncio
    async def test_openai_embed_with_native_format(self, mock_openai_client):
        """Test OpenAI embedding returns Neo4j-compatible vector."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIEmbeddingsProvider()
            
            # Get embedding with native format
            embedding = await provider.embed_text("test text", native_format=True)
            
            # Should return numpy array
            assert isinstance(embedding, np.ndarray)
            assert embedding.dtype == np.float32
            
            # Should be normalized for cosine similarity
            magnitude = np.linalg.norm(embedding)
            assert pytest.approx(magnitude, 0.0001) == 1.0
    
    @pytest.mark.asyncio
    async def test_openai_embed_backward_compatibility(self, mock_openai_client):
        """Test backward compatibility with list format."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIEmbeddingsProvider()
            
            # Default behavior should return list (backward compatible)
            embedding = await provider.embed_text("test text")
            
            assert isinstance(embedding, list)
            assert all(isinstance(x, float) for x in embedding)
    
    @pytest.mark.asyncio
    async def test_openai_batch_embed_with_native_format(self, mock_openai_client):
        """Test batch embedding with native Neo4j format."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIEmbeddingsProvider()
            
            # Mock batch response
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[0.1, 0.2, 0.3]),
                MagicMock(embedding=[0.4, 0.5, 0.6]),
                MagicMock(embedding=[0.7, 0.8, 0.9])
            ]
            mock_openai_client.embeddings.create.return_value = mock_response
            
            texts = ["text1", "text2", "text3"]
            embeddings = await provider.embed_batch(texts, native_format=True)
            
            assert len(embeddings) == 3
            for embedding in embeddings:
                assert isinstance(embedding, np.ndarray)
                assert embedding.dtype == np.float32
                # Check normalization
                magnitude = np.linalg.norm(embedding)
                assert pytest.approx(magnitude, 0.0001) == 1.0
    
    @pytest.mark.asyncio
    async def test_openai_dimension_validation(self, mock_openai_client):
        """Test dimension validation in OpenAI provider."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIEmbeddingsProvider(dimension=1536)
            
            # Mock response with wrong dimensions
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[0.1] * 1024)  # Wrong dimension
            ]
            mock_openai_client.embeddings.create.return_value = mock_response
            
            with pytest.raises(ValueError, match="dimension mismatch"):
                await provider.embed_text("test", native_format=True, validate_dimensions=True)


class TestVoyageNativeVectorSupport:
    """Test Voyage provider with native Neo4j vector support."""
    
    @pytest.fixture
    def mock_voyage_client(self):
        """Mock Voyage client for testing."""
        with patch("src.project_watch_mcp.utils.embeddings.voyage.voyageai.AsyncClient") as mock:
            client_instance = AsyncMock()
            mock.return_value = client_instance
            
            # Mock embedding response
            mock_response = MagicMock()
            mock_response.embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5]]
            client_instance.embed.return_value = mock_response
            
            yield client_instance
    
    @pytest.mark.asyncio
    async def test_voyage_embed_with_native_format(self, mock_voyage_client):
        """Test Voyage embedding returns Neo4j-compatible vector."""
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "test-key"}):
            provider = VoyageEmbeddingsProvider()
            
            # Get embedding with native format
            embedding = await provider.embed_text("test text", native_format=True)
            
            # Should return numpy array
            assert isinstance(embedding, np.ndarray)
            assert embedding.dtype == np.float32
            
            # Should be normalized for cosine similarity
            magnitude = np.linalg.norm(embedding)
            assert pytest.approx(magnitude, 0.0001) == 1.0
    
    @pytest.mark.asyncio
    async def test_voyage_batch_with_native_format(self, mock_voyage_client):
        """Test Voyage batch embedding with native format."""
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "test-key"}):
            provider = VoyageEmbeddingsProvider()
            
            # Mock batch response
            mock_response = MagicMock()
            mock_response.embeddings = [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9]
            ]
            mock_voyage_client.embed.return_value = mock_response
            
            texts = ["text1", "text2", "text3"]
            embeddings = await provider.embed_batch(texts, native_format=True)
            
            assert len(embeddings) == 3
            for embedding in embeddings:
                assert isinstance(embedding, np.ndarray)
                assert embedding.dtype == np.float32
                # Check normalization
                magnitude = np.linalg.norm(embedding)
                assert pytest.approx(magnitude, 0.0001) == 1.0
    
    @pytest.mark.asyncio
    async def test_voyage_dimension_auto_detection(self, mock_voyage_client):
        """Test Voyage provider auto-detects dimensions from model."""
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "test-key"}):
            # voyage-code-3 has 1024 dimensions
            provider = VoyageEmbeddingsProvider(model="voyage-code-3")
            assert provider.dimension == 1024
            
            # voyage-3-lite has 512 dimensions
            provider = VoyageEmbeddingsProvider(model="voyage-3-lite")
            assert provider.dimension == 512


class TestProviderFallbackMechanisms:
    """Test fallback mechanisms for embedding providers."""
    
    @pytest.mark.asyncio
    async def test_fallback_to_list_format_on_error(self):
        """Test providers fallback to list format if native format fails."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("src.project_watch_mcp.utils.embeddings.openai.AsyncOpenAI") as mock:
                client_instance = AsyncMock()
                mock.return_value = client_instance
                
                # Mock successful response
                mock_response = MagicMock()
                mock_response.data = [
                    MagicMock(embedding=[0.1, 0.2, 0.3])
                ]
                client_instance.embeddings.create.return_value = mock_response
                
                provider = OpenAIEmbeddingsProvider()
                
                # Simulate numpy conversion error
                with patch.object(provider, 'to_neo4j_vector', side_effect=ValueError("numpy error")):
                    # Should fallback to list format
                    embedding = await provider.embed_text("test", native_format=True)
                    assert isinstance(embedding, list)
    
    def test_cross_provider_vector_compatibility(self):
        """Test vectors from different providers are compatible."""
        # Create mock vectors from different providers
        openai_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        voyage_vector = np.array([0.4, 0.5, 0.6], dtype=np.float32)
        
        # Normalize both
        openai_norm = openai_vector / np.linalg.norm(openai_vector)
        voyage_norm = voyage_vector / np.linalg.norm(voyage_vector)
        
        # Both should be valid Neo4j vectors
        assert openai_norm.dtype == np.float32
        assert voyage_norm.dtype == np.float32
        assert pytest.approx(np.linalg.norm(openai_norm), 0.0001) == 1.0
        assert pytest.approx(np.linalg.norm(voyage_norm), 0.0001) == 1.0


class TestPerformanceOptimizations:
    """Test performance optimizations for native vector support."""
    
    @pytest.mark.asyncio
    async def test_batch_processing_efficiency(self):
        """Test batch processing is more efficient than individual calls."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("src.project_watch_mcp.utils.embeddings.openai.AsyncOpenAI") as mock:
                client_instance = AsyncMock()
                mock.return_value = client_instance
                
                # Mock batch response
                mock_response = MagicMock()
                mock_response.data = [
                    MagicMock(embedding=[0.1] * 1536) for _ in range(10)
                ]
                client_instance.embeddings.create.return_value = mock_response
                
                provider = OpenAIEmbeddingsProvider()
                texts = ["text" + str(i) for i in range(10)]
                
                # Batch processing should make single API call
                await provider.embed_batch(texts)
                assert client_instance.embeddings.create.call_count == 1
    
    def test_memory_efficient_vector_storage(self):
        """Test vectors use memory-efficient float32 format."""
        provider = EmbeddingsProvider()
        
        # Create large embedding
        large_embedding = [0.1] * 3072  # Large dimension
        
        # Convert to Neo4j format
        neo4j_vector = provider.to_neo4j_vector(large_embedding)
        
        # Should use float32 (4 bytes) instead of float64 (8 bytes)
        assert neo4j_vector.dtype == np.float32
        
        # Memory usage should be ~12KB instead of ~24KB
        memory_usage = neo4j_vector.nbytes
        assert memory_usage == 3072 * 4  # 4 bytes per float32