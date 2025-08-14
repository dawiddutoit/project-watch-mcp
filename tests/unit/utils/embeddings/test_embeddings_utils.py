"""Test utilities for embeddings."""

from project_watch_mcp.utils.embeddings import EmbeddingsProvider


class TestEmbeddingsProvider(EmbeddingsProvider):
    """Test embeddings provider for unit tests only."""

    def __init__(self, dimension: int = 1536):
        """Initialize test embeddings provider."""
        super().__init__(dimension)

    async def embed_text(self, text: str) -> list[float]:
        """Generate test embedding vector."""
        # Return a deterministic test embedding based on text length
        # This is ONLY for testing, not for production use
        import hashlib
        
        # Create a hash of the text for deterministic output
        hash_value = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        
        # Generate deterministic values
        embedding = []
        for i in range(self.dimension):
            # Generate a pseudo-random float between -1 and 1
            value = ((hash_value + i) % 1000) / 500.0 - 1.0
            embedding.append(value)
        
        return embedding