"""Unit tests for base embeddings module."""

import pytest

from project_watch_mcp.utils.embeddings.base import EmbeddingsProvider


class TestEmbeddingsProvider:
    """Test EmbeddingsProvider abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that EmbeddingsProvider cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            EmbeddingsProvider()

    def test_concrete_implementation_required(self):
        """Test that concrete implementations must implement abstract methods."""
        
        class IncompleteProvider(EmbeddingsProvider):
            """Incomplete provider missing embed method."""
            pass
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteProvider()

    def test_valid_concrete_implementation(self):
        """Test that a valid concrete implementation can be instantiated."""
        
        class ConcreteProvider(EmbeddingsProvider):
            """Complete provider with embed method."""
            
            async def embed(self, text: str) -> list[float]:
                """Implement embed method."""
                return [1.0, 2.0, 3.0]
        
        # Should not raise
        provider = ConcreteProvider()
        assert isinstance(provider, EmbeddingsProvider)

    @pytest.mark.asyncio
    async def test_embed_method_signature(self):
        """Test that embed method has correct signature."""
        
        class TestProvider(EmbeddingsProvider):
            """Test provider implementation."""
            
            async def embed(self, text: str) -> list[float]:
                """Implement embed method."""
                return [float(ord(c)) for c in text[:10]]
        
        provider = TestProvider()
        result = await provider.embed("test")
        
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)