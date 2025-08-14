"""Test real embeddings providers (requires API keys)."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from project_watch_mcp.utils.embeddings import (
    OpenAIEmbeddingsProvider,
    VoyageEmbeddingsProvider,
    create_embeddings_provider,
)


class TestEmbeddingsProviderFactory:
    """Test the embeddings provider factory."""

    def test_factory_returns_none_without_api_key(self):
        """Test that factory returns None when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            provider = create_embeddings_provider("openai")
            assert provider is None
            
            provider = create_embeddings_provider("voyage")
            assert provider is None

    def test_factory_creates_openai_with_api_key(self):
        """Test that factory creates OpenAI provider with API key."""
        with patch("project_watch_mcp.utils.embeddings.openai.AsyncOpenAI"):
            provider = create_embeddings_provider("openai", api_key="test-key")
            assert provider is not None
            assert isinstance(provider, OpenAIEmbeddingsProvider)

    def test_factory_creates_voyage_with_api_key(self):
        """Test that factory creates Voyage provider with API key."""
        with patch("project_watch_mcp.utils.embeddings.voyage.voyageai.AsyncClient"):
            provider = create_embeddings_provider("voyage", api_key="test-key")
            assert provider is not None
            assert isinstance(provider, VoyageEmbeddingsProvider)

    def test_factory_raises_for_invalid_provider(self):
        """Test that factory raises error for invalid provider type."""
        with pytest.raises(ValueError, match="Unknown provider type"):
            create_embeddings_provider("invalid")


class TestOpenAIEmbeddingsProvider:
    """Test OpenAI embeddings provider."""

    @pytest.mark.asyncio
    async def test_openai_embedding_dimensions(self):
        """Test that OpenAI returns correct embedding dimensions."""
        with patch("project_watch_mcp.utils.embeddings.openai.AsyncOpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Mock embedding response
            test_dimension = 1536
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1] * test_dimension)]
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            
            provider = OpenAIEmbeddingsProvider(api_key="test", dimension=test_dimension)
            embedding = await provider.embed_text("test")
            
            assert len(embedding) == test_dimension
            assert all(isinstance(x, float) for x in embedding)


class TestVoyageEmbeddingsProvider:
    """Test Voyage embeddings provider."""

    @pytest.mark.asyncio
    async def test_voyage_embedding_dimensions(self):
        """Test that Voyage returns correct embedding dimensions."""
        with patch("project_watch_mcp.utils.embeddings.voyage.voyageai.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Mock embedding response (voyage-code-3 has 1024 dimensions)
            test_dimension = 1024
            mock_response = MagicMock()
            mock_response.embeddings = [[0.2] * test_dimension]
            mock_client.embed = AsyncMock(return_value=mock_response)
            
            provider = VoyageEmbeddingsProvider(api_key="test", model="voyage-code-3")
            embedding = await provider.embed_text("test")
            
            assert len(embedding) == test_dimension
            assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_voyage_document_vs_query_embeddings(self):
        """Test that Voyage handles document vs query types."""
        with patch("project_watch_mcp.utils.embeddings.voyage.voyageai.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            call_count = 0
            input_types = []
            
            async def track_calls(texts, model, input_type):
                nonlocal call_count
                call_count += 1
                input_types.append(input_type)
                return MagicMock(embeddings=[[0.1] * 1024])
            
            mock_client.embed = AsyncMock(side_effect=track_calls)
            
            provider = VoyageEmbeddingsProvider(api_key="test")
            
            # Test document embedding
            await provider.embed_text("document text", input_type="document")
            assert input_types[0] == "document"
            
            # Test query embedding  
            await provider.embed_text("query text", input_type="query")
            assert input_types[1] == "query"
            
            assert call_count == 2


class TestEmbeddingAccuracy:
    """Test embedding accuracy and behavior."""

    @pytest.mark.asyncio
    async def test_same_text_produces_same_embedding(self):
        """Test that same text produces same embedding."""
        with patch("project_watch_mcp.utils.embeddings.openai.AsyncOpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Return same embedding for same input
            fixed_embedding = [0.5] * 1536
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=fixed_embedding)]
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            
            provider = OpenAIEmbeddingsProvider(api_key="test")
            
            embedding1 = await provider.embed_text("test")
            embedding2 = await provider.embed_text("test")
            
            assert embedding1 == embedding2

    @pytest.mark.asyncio
    async def test_text_truncation(self):
        """Test that long text is properly truncated."""
        with patch("project_watch_mcp.utils.embeddings.openai.AsyncOpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            truncated_text = None
            async def capture_text(input, model):
                nonlocal truncated_text
                truncated_text = input
                return MagicMock(data=[MagicMock(embedding=[0.1] * 10)])
            
            mock_client.embeddings.create = AsyncMock(side_effect=capture_text)
            
            provider = OpenAIEmbeddingsProvider(api_key="test", max_tokens=5)
            
            with patch.object(provider.tokenizer, "encode") as mock_encode:
                with patch.object(provider.tokenizer, "decode") as mock_decode:
                    mock_encode.return_value = list(range(100))  # More than max_tokens
                    mock_decode.return_value = "truncated"
                    
                    await provider.embed_text("very long text")
                    
                    assert truncated_text == "truncated"