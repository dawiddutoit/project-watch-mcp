"""Unit tests for Voyage embeddings provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from project_watch_mcp.utils.embeddings.voyage import VoyageEmbeddingsProvider


class TestVoyageEmbeddingsProvider:
    """Test VoyageEmbeddingsProvider class."""

    @pytest.fixture
    def mock_voyage_client(self):
        """Create mock Voyage client."""
        mock = MagicMock()
        mock.embed = AsyncMock()
        return mock

    @pytest.fixture
    def provider(self, mock_voyage_client):
        """Create VoyageEmbeddingsProvider with mocked client."""
        with patch("project_watch_mcp.utils.embeddings.voyage.voyageai.AsyncClient", return_value=mock_voyage_client):
            provider = VoyageEmbeddingsProvider(api_key="test-key", model="voyage-code-2")
            return provider

    def test_initialization(self):
        """Test provider initialization."""
        with patch("project_watch_mcp.utils.embeddings.voyage.voyageai.AsyncClient") as mock_client:
            provider = VoyageEmbeddingsProvider(api_key="test-key", model="custom-model")
            
            mock_client.assert_called_once_with(api_key="test-key")
            assert provider.model == "custom-model"

    def test_default_model(self):
        """Test default model is used when not specified."""
        with patch("project_watch_mcp.utils.embeddings.voyage.voyageai.AsyncClient"):
            provider = VoyageEmbeddingsProvider(api_key="test-key")
            assert provider.model == "voyage-code-3"

    @pytest.mark.asyncio
    async def test_embed_success(self, provider, mock_voyage_client):
        """Test successful embedding generation."""
        # Mock response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5]]
        mock_voyage_client.embed.return_value = mock_response
        
        # Generate embedding
        result = await provider.embed("test code")
        
        # Verify API call
        mock_voyage_client.embed.assert_called_once_with(
            texts=["test code"],
            model="voyage-code-2",
            input_type="document"
        )
        
        # Verify result
        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]

    @pytest.mark.asyncio
    async def test_embed_empty_text(self, provider, mock_voyage_client):
        """Test embedding generation with empty text."""
        mock_response = MagicMock()
        mock_response.embeddings = [[0.0] * 1536]
        mock_voyage_client.embed.return_value = mock_response
        
        result = await provider.embed("")
        
        mock_voyage_client.embed.assert_called_once_with(
            texts=[""],
            model="voyage-code-2",
            input_type="document"
        )
        assert len(result) == 1536

    @pytest.mark.asyncio
    async def test_embed_code_snippet(self, provider, mock_voyage_client):
        """Test embedding generation with code snippet."""
        code_snippet = """
def hello_world():
    print("Hello, World!")
    return 42
"""
        
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1024]
        mock_voyage_client.embed.return_value = mock_response
        
        result = await provider.embed(code_snippet)
        
        mock_voyage_client.embed.assert_called_once_with(
            texts=[code_snippet],
            model="voyage-code-2",
            input_type="document"
        )
        assert len(result) == 1024

    @pytest.mark.asyncio
    async def test_embed_api_error(self, provider, mock_voyage_client):
        """Test handling of API errors."""
        mock_voyage_client.embed.side_effect = Exception("Voyage API Error")
        
        with pytest.raises(Exception, match="Voyage API Error"):
            await provider.embed("test code")

    @pytest.mark.asyncio
    async def test_embed_with_input_type(self, provider, mock_voyage_client):
        """Test that input_type is always set to document."""
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2]]
        mock_voyage_client.embed.return_value = mock_response
        
        await provider.embed("query text")
        
        # Verify input_type is always "document" for consistency
        mock_voyage_client.embed.assert_called_once_with(
            texts=["query text"],
            model="voyage-code-2",
            input_type="document"
        )

    @pytest.mark.asyncio
    async def test_embed_batch_handling(self, provider, mock_voyage_client):
        """Test that single text is wrapped in list for batch API."""
        mock_response = MagicMock()
        mock_response.embeddings = [[0.5, 0.6, 0.7]]
        mock_voyage_client.embed.return_value = mock_response
        
        result = await provider.embed("single text")
        
        # Verify text is wrapped in list
        mock_voyage_client.embed.assert_called_once_with(
            texts=["single text"],
            model="voyage-code-2",
            input_type="document"
        )
        # Verify single embedding is extracted
        assert result == [0.5, 0.6, 0.7]

    @pytest.mark.asyncio
    async def test_embed_preserves_dimension(self, provider, mock_voyage_client):
        """Test that embedding dimension is preserved."""
        # Test with different dimensions
        for dimension in [512, 1024, 1536, 2048]:
            mock_response = MagicMock()
            mock_response.embeddings = [[0.1] * dimension]
            mock_voyage_client.embed.return_value = mock_response
            
            result = await provider.embed(f"test {dimension}")
            assert len(result) == dimension