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
        with patch(
            "project_watch_mcp.utils.embeddings.voyage.voyageai.AsyncClient",
            return_value=mock_voyage_client,
        ):
            provider = VoyageEmbeddingsProvider(api_key="test-key", model="voyage-code-3")
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
            texts=["test code"], model="voyage-code-3", input_type="document"
        )

        # Verify result
        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]

    @pytest.mark.asyncio
    async def test_embed_empty_text(self, provider, mock_voyage_client):
        """Test embedding generation with empty text returns zero vector without API call."""
        # Should not call API for empty text
        result = await provider.embed("")

        # API should NOT be called for empty strings
        mock_voyage_client.embed.assert_not_called()

        # Should return zero vector with correct dimensions
        assert len(result) == 1024  # voyage-code-3 default dimension
        assert all(v == 0.0 for v in result)

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
            texts=[code_snippet], model="voyage-code-3", input_type="document"
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
            texts=["query text"], model="voyage-code-3", input_type="document"
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
            texts=["single text"], model="voyage-code-3", input_type="document"
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

    @pytest.mark.asyncio
    async def test_embed_text_empty_string(self, provider, mock_voyage_client):
        """Test embed_text with empty string returns zero vector."""
        result = await provider.embed_text("")

        # Should not call API
        mock_voyage_client.embed.assert_not_called()

        # Should return zero vector with correct dimensions
        assert len(result) == 1024
        assert all(v == 0.0 for v in result)

    @pytest.mark.asyncio
    async def test_embed_text_whitespace_only(self, provider, mock_voyage_client):
        """Test embed_text with whitespace-only string returns zero vector."""
        # Test various whitespace strings
        whitespace_strings = ["   ", "\t", "\n", "  \n\t  ", "\r\n"]

        for ws in whitespace_strings:
            mock_voyage_client.embed.reset_mock()
            result = await provider.embed_text(ws)

            # Should not call API for whitespace-only strings
            mock_voyage_client.embed.assert_not_called()

            # Should return zero vector
            assert len(result) == 1024
            assert all(v == 0.0 for v in result), f"Failed for whitespace: {repr(ws)}"

    @pytest.mark.asyncio
    async def test_embed_batch_with_empty_strings(self, provider, mock_voyage_client):
        """Test embed_batch filters out empty strings and returns zero vectors."""
        texts = ["valid text 1", "", "valid text 2", "   ", "valid text 3"]

        # Mock response for non-empty texts only
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
        mock_voyage_client.embed.return_value = mock_response

        result = await provider.embed_batch(texts)

        # Should call API with only non-empty texts
        mock_voyage_client.embed.assert_called_once_with(
            texts=["valid text 1", "valid text 2", "valid text 3"],
            model="voyage-code-3",
            input_type="document",
        )

        # Should return 5 embeddings (matching input count)
        assert len(result) == 5

        # Check embeddings match expected positions
        assert result[0] == [0.1] * 1024  # valid text 1
        assert all(v == 0.0 for v in result[1])  # empty string
        assert result[2] == [0.2] * 1024  # valid text 2
        assert all(v == 0.0 for v in result[3])  # whitespace
        assert result[4] == [0.3] * 1024  # valid text 3

    @pytest.mark.asyncio
    async def test_embed_batch_all_empty_strings(self, provider, mock_voyage_client):
        """Test embed_batch with all empty strings returns zero vectors without API call."""
        texts = ["", "   ", "\t", "\n"]

        result = await provider.embed_batch(texts)

        # Should not call API when all strings are empty
        mock_voyage_client.embed.assert_not_called()

        # Should return zero vectors for all inputs
        assert len(result) == 4
        for embedding in result:
            assert len(embedding) == 1024
            assert all(v == 0.0 for v in embedding)

    @pytest.mark.asyncio
    async def test_embed_text_with_native_format_empty_string(self, provider, mock_voyage_client):
        """Test embed_text with native_format=True for empty string."""
        import numpy as np

        result = await provider.embed_text("", native_format=True)

        # Should not call API
        mock_voyage_client.embed.assert_not_called()

        # Should return numpy array of zeros
        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)
        assert np.all(result == 0.0)

    @pytest.mark.asyncio
    async def test_embed_batch_with_native_format_mixed(self, provider, mock_voyage_client):
        """Test embed_batch with native_format=True for mixed empty/non-empty texts."""
        import numpy as np

        texts = ["valid text", "", "   "]

        # Mock response for non-empty text only
        mock_response = MagicMock()
        mock_response.embeddings = [[0.5] * 1024]
        mock_voyage_client.embed.return_value = mock_response

        result = await provider.embed_batch(texts, native_format=True)

        # Check results
        assert len(result) == 3

        # First should be normalized non-zero vector
        assert isinstance(result[0], np.ndarray)
        assert result[0].shape == (1024,)

        # Second and third should be zero vectors
        for i in [1, 2]:
            assert isinstance(result[i], np.ndarray)
            assert result[i].shape == (1024,)
            assert np.all(result[i] == 0.0)

    def test_dimension_for_different_models(self):
        """Test that different models have correct dimensions."""
        with patch("project_watch_mcp.utils.embeddings.voyage.voyageai.AsyncClient"):
            # Test voyage-code-3
            provider = VoyageEmbeddingsProvider(api_key="test-key", model="voyage-code-3")
            assert provider.dimension == 1024

            # Test voyage-3
            provider = VoyageEmbeddingsProvider(api_key="test-key", model="voyage-3")
            assert provider.dimension == 1024

            # Test voyage-3-lite
            provider = VoyageEmbeddingsProvider(api_key="test-key", model="voyage-3-lite")
            assert provider.dimension == 512
