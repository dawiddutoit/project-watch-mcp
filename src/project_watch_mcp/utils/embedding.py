"""Embedding providers for semantic code search."""

import logging
import os
from typing import Literal

import httpx

logger = logging.getLogger(__name__)


class EmbeddingsProvider:
    """Base embeddings provider interface."""

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for text."""
        raise NotImplementedError

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        raise NotImplementedError


class OpenAIEmbeddingsProvider(EmbeddingsProvider):
    """OpenAI embeddings provider using text-embedding-3-small model."""

    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small"):
        """Initialize OpenAI embeddings provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: OpenAI embedding model to use
        """
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError("OpenAI library not installed. Run: uv add openai") from e

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable."
            )

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model
        self._dimension = 1536 if "text-embedding-3-small" in model else 3072  # ada-002 vs large

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for text using OpenAI."""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate OpenAI embedding: {e}")
            raise

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Failed to generate OpenAI embeddings batch: {e}")
            raise

    @property
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        return self._dimension


class LocalEmbeddingsProvider(EmbeddingsProvider):
    """Local embeddings provider using a lightweight model via HTTP API."""

    def __init__(self, api_url: str = "http://localhost:8080/embeddings", dimension: int = 384):
        """Initialize local embeddings provider.

        Args:
            api_url: URL of the local embeddings API
            dimension: Dimension of the embeddings
        """
        self.api_url = api_url
        self._dimension = dimension
        self.client = httpx.AsyncClient(timeout=30.0)

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for text using local model."""
        try:
            response = await self.client.post(
                self.api_url,
                json={"text": text},
            )
            response.raise_for_status()
            data = response.json()
            return data["embedding"]
        except httpx.ConnectError as e:
            raise ConnectionError(
                f"Local embeddings API not available at {self.api_url}. "
                f"Please ensure your embedding server is running or switch to a different provider."
            ) from e
        except Exception as e:
            logger.error(f"Failed to generate local embedding: {e}")
            raise

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        try:
            response = await self.client.post(
                self.api_url,
                json={"texts": texts},
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]
        except httpx.ConnectError as e:
            raise ConnectionError(
                f"Local embeddings API not available at {self.api_url}. "
                f"Please ensure your embedding server is running or switch to a different provider."
            ) from e
        except Exception as e:
            logger.error(f"Failed to generate local embeddings batch: {e}")
            raise

    @property
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        return self._dimension

    async def __aexit__(self, *args):
        """Clean up the HTTP client."""
        await self.client.aclose()


class MockEmbeddingsProvider(EmbeddingsProvider):
    """Mock embeddings provider for testing."""

    def __init__(self, dimension: int = 384):
        """Initialize mock embeddings provider.

        Args:
            dimension: Dimension of the mock embeddings
        """
        self._dimension = dimension

    async def embed_text(self, text: str) -> list[float]:
        """Generate mock embedding for text."""
        # Generate deterministic mock embedding based on text hash
        hash_val = hash(text)
        base_val = (hash_val % 100) / 100.0
        return [base_val + (i * 0.001) for i in range(self._dimension)]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings for multiple texts."""
        return [await self.embed_text(text) for text in texts]

    @property
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        return self._dimension


def create_embeddings_provider(
    provider_type: Literal["openai", "local", "mock"] = "mock", **kwargs
) -> EmbeddingsProvider:
    """Factory function to create embeddings provider.

    Args:
        provider_type: Type of embeddings provider to create
        **kwargs: Additional arguments for the provider

    Returns:
        EmbeddingsProvider instance
    """
    if provider_type == "openai":
        return OpenAIEmbeddingsProvider(**kwargs)
    elif provider_type == "local":
        # Filter kwargs for LocalEmbeddingsProvider
        local_kwargs = {k: v for k, v in kwargs.items() if k in ["api_url", "dimension"]}
        return LocalEmbeddingsProvider(**local_kwargs)
    elif provider_type == "mock":
        # Filter kwargs for MockEmbeddingsProvider
        mock_kwargs = {k: v for k, v in kwargs.items() if k in ["dimension"]}
        return MockEmbeddingsProvider(**mock_kwargs)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
