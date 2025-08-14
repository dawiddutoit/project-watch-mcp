"""Test suite for Voyage AI embeddings provider."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from project_watch_mcp.utils.embeddings import VoyageEmbeddingsProvider


@pytest.fixture
def voyage_api_key():
    """Fixture to provide test API key."""
    return "test-voyage-api-key"


@pytest.fixture
def mock_voyage_client():
    """Mock Voyage client for testing."""
    with patch("voyageai.AsyncClient") as mock_async_client:
        mock_client = MagicMock()
        mock_async_client.return_value = mock_client
        yield mock_client


class TestVoyageEmbeddingsProvider:
    """Test cases for VoyageEmbeddingsProvider."""

    def test_init_with_api_key(self, voyage_api_key):
        """Test initialization with explicit API key."""
        provider = VoyageEmbeddingsProvider(api_key=voyage_api_key)
        assert provider.api_key == voyage_api_key
        assert provider.model == "voyage-code-3"
        assert provider.dimension == 1024  # voyage-code-3 has 1024 dimensions

    def test_init_with_env_var(self, voyage_api_key):
        """Test initialization with environment variable."""
        with patch.dict(os.environ, {"VOYAGE_API_KEY": voyage_api_key}):
            provider = VoyageEmbeddingsProvider()
            assert provider.api_key == voyage_api_key

    def test_init_without_api_key(self):
        """Test initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Voyage API key not found"):
                VoyageEmbeddingsProvider()

    def test_different_model_dimensions(self):
        """Test different Voyage models have correct dimensions."""
        with patch.dict(os.environ, {"VOYAGE_API_KEY": "test-key"}):
            # voyage-code-3 (default for code)
            provider = VoyageEmbeddingsProvider(model="voyage-code-3")
            assert provider.dimension == 1024

            # voyage-3 (general purpose)
            provider = VoyageEmbeddingsProvider(model="voyage-3")
            assert provider.dimension == 1024

            # voyage-3-lite (lightweight)
            provider = VoyageEmbeddingsProvider(model="voyage-3-lite")
            assert provider.dimension == 512

    @pytest.mark.asyncio
    async def test_embed_text_success(self, voyage_api_key, mock_voyage_client):
        """Test successful text embedding."""
        # Setup mock response
        expected_embedding = [0.1] * 1024
        mock_response = MagicMock()
        mock_response.embeddings = [expected_embedding]
        mock_voyage_client.embed = AsyncMock(return_value=mock_response)

        provider = VoyageEmbeddingsProvider(api_key=voyage_api_key)
        result = await provider.embed_text("test code snippet")

        assert result == expected_embedding
        mock_voyage_client.embed.assert_called_once_with(
            texts=["test code snippet"], model="voyage-code-3", input_type="document"
        )

    @pytest.mark.asyncio
    async def test_embed_text_with_query_input_type(self, voyage_api_key, mock_voyage_client):
        """Test embedding with query input type."""
        expected_embedding = [0.2] * 1024
        mock_response = MagicMock()
        mock_response.embeddings = [expected_embedding]
        mock_voyage_client.embed = AsyncMock(return_value=mock_response)

        provider = VoyageEmbeddingsProvider(api_key=voyage_api_key)
        result = await provider.embed_text("search query", input_type="query")

        assert result == expected_embedding
        mock_voyage_client.embed.assert_called_once_with(
            texts=["search query"], model="voyage-code-3", input_type="query"
        )

    @pytest.mark.asyncio
    async def test_embed_text_truncation(self, voyage_api_key, mock_voyage_client):
        """Test text truncation when exceeding max tokens."""
        # Create a very long text (Voyage has context limit of 32000 tokens)
        long_text = "code " * 40000  # Exceeds token limit

        expected_embedding = [0.3] * 1024
        mock_response = MagicMock()
        mock_response.embeddings = [expected_embedding]
        mock_voyage_client.embed = AsyncMock(return_value=mock_response)

        provider = VoyageEmbeddingsProvider(api_key=voyage_api_key)
        with patch.object(provider, "_truncate_text") as mock_truncate:
            mock_truncate.return_value = "truncated text"
            result = await provider.embed_text(long_text)

            assert result == expected_embedding
            mock_truncate.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_text_error_handling(self, voyage_api_key, mock_voyage_client):
        """Test error handling during embedding."""
        mock_voyage_client.embed = AsyncMock(side_effect=Exception("API error"))

        provider = VoyageEmbeddingsProvider(api_key=voyage_api_key)
        with pytest.raises(Exception, match="API error"):
            await provider.embed_text("test text")

    @pytest.mark.asyncio
    async def test_embed_batch_texts(self, voyage_api_key, mock_voyage_client):
        """Test batch text embedding."""
        texts = ["code1", "code2", "code3"]
        expected_embeddings = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
        mock_response = MagicMock()
        mock_response.embeddings = expected_embeddings
        mock_voyage_client.embed = AsyncMock(return_value=mock_response)

        provider = VoyageEmbeddingsProvider(api_key=voyage_api_key)
        result = await provider.embed_batch(texts)

        assert result == expected_embeddings
        mock_voyage_client.embed.assert_called_once_with(
            texts=texts, model="voyage-code-3", input_type="document"
        )

    def test_provider_compatibility_interface(self, voyage_api_key):
        """Test that VoyageEmbeddingsProvider follows the same interface as OpenAIEmbeddingsProvider."""
        from project_watch_mcp.utils.embeddings import EmbeddingsProvider

        provider = VoyageEmbeddingsProvider(api_key=voyage_api_key)
        assert isinstance(provider, EmbeddingsProvider)
        assert hasattr(provider, "dimension")
        assert hasattr(provider, "embed_text")
        assert provider.dimension == 1024  # Different from OpenAI's 1536
