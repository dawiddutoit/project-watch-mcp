"""Unit tests for embeddings module with mocked APIs."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from project_watch_mcp.utils.embeddings import (
    EmbeddingsProvider,
    OpenAIEmbeddingsProvider,
    VoyageEmbeddingsProvider,
    create_embeddings_provider,
)


class TestEmbeddingsProvider:
    """Test the base EmbeddingsProvider class."""

    def test_base_initialization(self):
        """Test base class initialization."""
        provider = EmbeddingsProvider(dimension=768)
        assert provider.dimension == 768

    def test_base_default_dimension(self):
        """Test default dimension."""
        provider = EmbeddingsProvider()
        assert provider.dimension == 1536

    @pytest.mark.asyncio
    async def test_base_embed_text_not_implemented(self):
        """Test that base class raises NotImplementedError."""
        provider = EmbeddingsProvider()
        with pytest.raises(NotImplementedError):
            await provider.embed_text("test")


class TestEmbeddingsFactory:
    """Test the create_embeddings_provider factory function."""

    def test_factory_returns_none_without_api_key(self):
        """Test factory returns None when API key missing."""
        with patch.dict(os.environ, {}, clear=True):
            provider = create_embeddings_provider("openai")
            assert provider is None

    def test_factory_creates_openai_provider(self):
        """Test factory creates OpenAI provider with API key."""
        with patch("project_watch_mcp.utils.embeddings.openai.AsyncOpenAI"):
            provider = create_embeddings_provider(
                "openai",
                api_key="test-key",
                model="text-embedding-3-small"
            )
            assert isinstance(provider, OpenAIEmbeddingsProvider)
            assert provider.api_key == "test-key"
            assert provider.model == "text-embedding-3-small"

    def test_factory_creates_voyage_provider(self):
        """Test factory creates Voyage provider with API key."""
        with patch("project_watch_mcp.utils.embeddings.voyage.voyageai.AsyncClient"):
            provider = create_embeddings_provider(
                "voyage",
                api_key="test-key",
                model="voyage-code-3"
            )
            assert isinstance(provider, VoyageEmbeddingsProvider)
            assert provider.api_key == "test-key"
            assert provider.model == "voyage-code-3"

    def test_factory_invalid_provider_type(self):
        """Test factory raises error for invalid provider."""
        with pytest.raises(ValueError, match="Unknown provider type"):
            create_embeddings_provider("invalid")

    def test_factory_filters_kwargs(self):
        """Test factory filters kwargs appropriately."""
        with patch("project_watch_mcp.utils.embeddings.openai.AsyncOpenAI"):
            # OpenAI should accept dimension
            provider = create_embeddings_provider(
                "openai",
                api_key="test",
                dimension=3072,
                invalid_param="ignored"
            )
            assert provider.dimension == 3072

    def test_factory_case_insensitive(self):
        """Test provider type is case insensitive."""
        with patch("project_watch_mcp.utils.embeddings.openai.AsyncOpenAI"):
            provider1 = create_embeddings_provider("OPENAI", api_key="test")
            provider2 = create_embeddings_provider("OpenAI", api_key="test")
            
            assert isinstance(provider1, OpenAIEmbeddingsProvider)
            assert isinstance(provider2, OpenAIEmbeddingsProvider)


class TestOpenAIEmbeddingsProvider:
    """Unit tests for OpenAI embeddings provider."""

    def test_initialization_with_api_key(self):
        """Test initialization with API key."""
        with patch("project_watch_mcp.utils.embeddings.openai.AsyncOpenAI"):
            provider = OpenAIEmbeddingsProvider(
                api_key="test-key",
                model="text-embedding-3-large",
                dimension=3072
            )
            assert provider.api_key == "test-key"
            assert provider.model == "text-embedding-3-large"
            assert provider.dimension == 3072

    def test_initialization_from_env(self):
        """Test initialization from environment variable."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            with patch("project_watch_mcp.utils.embeddings.openai.AsyncOpenAI"):
                provider = OpenAIEmbeddingsProvider()
                assert provider.api_key == "env-key"

    def test_no_api_key_raises_error(self):
        """Test missing API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key not found"):
                OpenAIEmbeddingsProvider()

    @pytest.mark.asyncio
    async def test_embed_text_success(self):
        """Test successful text embedding."""
        mock_embedding = [0.1, 0.2, 0.3] * 512  # 1536 dimensions
        
        with patch("project_watch_mcp.utils.embeddings.openai.AsyncOpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=mock_embedding)]
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            
            provider = OpenAIEmbeddingsProvider(api_key="test")
            result = await provider.embed_text("test text")
            
            assert result == mock_embedding
            mock_client.embeddings.create.assert_called_once_with(
                input="test text",
                model="text-embedding-3-small"
            )

    @pytest.mark.asyncio
    async def test_text_truncation(self):
        """Test text truncation for long inputs."""
        with patch("project_watch_mcp.utils.embeddings.openai.AsyncOpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1])]
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            
            provider = OpenAIEmbeddingsProvider(api_key="test", max_tokens=5)
            
            with patch.object(provider.tokenizer, "encode") as mock_encode:
                with patch.object(provider.tokenizer, "decode") as mock_decode:
                    mock_encode.return_value = list(range(10))  # More than max_tokens
                    mock_decode.return_value = "truncated"
                    
                    await provider.embed_text("very long text")
                    
                    mock_decode.assert_called_once()
                    mock_client.embeddings.create.assert_called_with(
                        input="truncated",
                        model="text-embedding-3-small"
                    )

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in embeddings."""
        with patch("project_watch_mcp.utils.embeddings.openai.AsyncOpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.embeddings.create = AsyncMock(side_effect=Exception("API Error"))
            
            provider = OpenAIEmbeddingsProvider(api_key="test")
            
            with pytest.raises(Exception, match="API Error"):
                await provider.embed_text("test")


class TestVoyageEmbeddingsProvider:
    """Unit tests for Voyage embeddings provider."""

    def test_initialization_with_api_key(self):
        """Test initialization with API key."""
        with patch("project_watch_mcp.utils.embeddings.voyage.voyageai.AsyncClient"):
            provider = VoyageEmbeddingsProvider(
                api_key="voyage-key",
                model="voyage-code-3"
            )
            assert provider.api_key == "voyage-key"
            assert provider.model == "voyage-code-3"
            assert provider.dimension == 1024  # voyage-code-3 dimension

    def test_model_dimensions(self):
        """Test correct dimensions for different models."""
        with patch("project_watch_mcp.utils.embeddings.voyage.voyageai.AsyncClient"):
            provider1 = VoyageEmbeddingsProvider(api_key="key", model="voyage-code-3")
            assert provider1.dimension == 1024
            
            provider2 = VoyageEmbeddingsProvider(api_key="key", model="voyage-3")
            assert provider2.dimension == 1024
            
            provider3 = VoyageEmbeddingsProvider(api_key="key", model="voyage-3-lite")
            assert provider3.dimension == 512

    def test_no_api_key_raises_error(self):
        """Test missing API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Voyage API key not found"):
                VoyageEmbeddingsProvider()

    @pytest.mark.asyncio
    async def test_embed_text_document(self):
        """Test document embedding."""
        mock_embedding = [0.1] * 1024
        
        with patch("project_watch_mcp.utils.embeddings.voyage.voyageai.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.embeddings = [mock_embedding]
            mock_client.embed = AsyncMock(return_value=mock_response)
            
            provider = VoyageEmbeddingsProvider(api_key="test")
            result = await provider.embed_text("test", input_type="document")
            
            assert result == mock_embedding
            mock_client.embed.assert_called_once_with(
                texts=["test"],
                model="voyage-code-3",
                input_type="document"
            )

    @pytest.mark.asyncio
    async def test_embed_text_query(self):
        """Test query embedding."""
        mock_embedding = [0.2] * 1024
        
        with patch("project_watch_mcp.utils.embeddings.voyage.voyageai.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.embeddings = [mock_embedding]
            mock_client.embed = AsyncMock(return_value=mock_response)
            
            provider = VoyageEmbeddingsProvider(api_key="test")
            result = await provider.embed_text("query", input_type="query")
            
            assert result == mock_embedding
            mock_client.embed.assert_called_once_with(
                texts=["query"],
                model="voyage-code-3",
                input_type="query"
            )

    @pytest.mark.asyncio
    async def test_text_truncation(self):
        """Test text truncation for long inputs."""
        with patch("project_watch_mcp.utils.embeddings.voyage.voyageai.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.embeddings = [[0.1]]
            mock_client.embed = AsyncMock(return_value=mock_response)
            
            provider = VoyageEmbeddingsProvider(api_key="test", max_tokens=10)
            
            # Text longer than max_tokens * 3.5 chars
            long_text = "x" * 100
            await provider.embed_text(long_text)
            
            # Verify text was truncated
            called_text = mock_client.embed.call_args[1]["texts"][0]
            assert len(called_text) == 35  # 10 * 3.5

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        """Test batch embedding."""
        mock_embeddings = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
        
        with patch("project_watch_mcp.utils.embeddings.voyage.voyageai.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.embeddings = mock_embeddings
            mock_client.embed = AsyncMock(return_value=mock_response)
            
            provider = VoyageEmbeddingsProvider(api_key="test")
            texts = ["text1", "text2", "text3"]
            result = await provider.embed_batch(texts)
            
            assert result == mock_embeddings
            assert len(result) == 3
            mock_client.embed.assert_called_once_with(
                texts=texts,
                model="voyage-code-3",
                input_type="document"
            )

    @pytest.mark.asyncio
    async def test_batch_error_handling(self):
        """Test error handling in batch embedding."""
        with patch("project_watch_mcp.utils.embeddings.voyage.voyageai.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.embed = AsyncMock(side_effect=Exception("Batch Error"))
            
            provider = VoyageEmbeddingsProvider(api_key="test")
            
            with pytest.raises(Exception, match="Batch Error"):
                await provider.embed_batch(["text1", "text2"])

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in single embedding."""
        with patch("project_watch_mcp.utils.embeddings.voyage.voyageai.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.embed = AsyncMock(side_effect=Exception("API Error"))
            
            provider = VoyageEmbeddingsProvider(api_key="test")
            
            with pytest.raises(Exception, match="API Error"):
                await provider.embed_text("test")