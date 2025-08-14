"""Unit tests for OpenAI embeddings provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from project_watch_mcp.utils.embeddings.openai import OpenAIEmbeddingsProvider


class TestOpenAIEmbeddingsProvider:
    """Test OpenAIEmbeddingsProvider class."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create mock OpenAI client."""
        mock = MagicMock()
        mock.embeddings = MagicMock()
        mock.embeddings.create = AsyncMock()
        return mock

    @pytest.fixture
    def provider(self, mock_openai_client):
        """Create OpenAIEmbeddingsProvider with mocked client."""
        with patch("project_watch_mcp.utils.embeddings.openai.AsyncOpenAI", return_value=mock_openai_client):
            provider = OpenAIEmbeddingsProvider(api_key="test-key", model="text-embedding-3-small")
            return provider

    def test_initialization(self):
        """Test provider initialization."""
        with patch("project_watch_mcp.utils.embeddings.openai.AsyncOpenAI") as mock_client:
            provider = OpenAIEmbeddingsProvider(api_key="test-key", model="custom-model")
            
            mock_client.assert_called_once_with(api_key="test-key")
            assert provider.model == "custom-model"

    def test_default_model(self):
        """Test default model is used when not specified."""
        with patch("project_watch_mcp.utils.embeddings.openai.AsyncOpenAI"):
            provider = OpenAIEmbeddingsProvider(api_key="test-key")
            assert provider.model == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_embed_success(self, provider, mock_openai_client):
        """Test successful embedding generation."""
        # Mock response
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_openai_client.embeddings.create.return_value = mock_response
        
        # Generate embedding
        result = await provider.embed("test text")
        
        # Verify API call
        mock_openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input="test text"
        )
        
        # Verify result
        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]

    @pytest.mark.asyncio
    async def test_embed_empty_text(self, provider, mock_openai_client):
        """Test embedding generation with empty text."""
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.0] * 1536  # Default dimension
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_openai_client.embeddings.create.return_value = mock_response
        
        result = await provider.embed("")
        
        mock_openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=""
        )
        assert len(result) == 1536

    @pytest.mark.asyncio
    async def test_embed_long_text(self, provider, mock_openai_client):
        """Test embedding generation with long text."""
        long_text = "test " * 1000  # Long text
        
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_openai_client.embeddings.create.return_value = mock_response
        
        result = await provider.embed(long_text)
        
        mock_openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=long_text
        )
        assert len(result) == 1536

    @pytest.mark.asyncio
    async def test_embed_api_error(self, provider, mock_openai_client):
        """Test handling of API errors."""
        mock_openai_client.embeddings.create.side_effect = Exception("API Error")
        
        with pytest.raises(Exception, match="API Error"):
            await provider.embed("test text")

    @pytest.mark.asyncio
    async def test_embed_with_special_characters(self, provider, mock_openai_client):
        """Test embedding generation with special characters."""
        special_text = "Test with émojis 😀 and symbols @#$%"
        
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_openai_client.embeddings.create.return_value = mock_response
        
        result = await provider.embed(special_text)
        
        mock_openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=special_text
        )
        assert result == [0.1, 0.2, 0.3]