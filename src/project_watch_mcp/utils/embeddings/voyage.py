"""Voyage AI embeddings provider optimized for code search with native Neo4j vector support."""

import logging
import os
from typing import Literal

import numpy as np
import voyageai

from .base import EmbeddingsProvider

logger = logging.getLogger(__name__)


class VoyageEmbeddingsProvider(EmbeddingsProvider):
    """Voyage AI embeddings provider optimized for code search."""

    # Model dimensions mapping
    MODEL_DIMENSIONS = {
        "voyage-code-3": 1024,
        "voyage-3": 1024,
        "voyage-3-lite": 512,
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "voyage-code-3",
        max_tokens: int = 32000,
    ):
        """
        Initialize Voyage AI Embeddings provider.

        Args:
            api_key: Voyage API key (defaults to environment variable)
            model: Voyage embedding model to use (voyage-code-3 is optimized for code)
            max_tokens: Maximum tokens allowed by the model
        """
        # Get dimension from model name
        dimension = self.MODEL_DIMENSIONS.get(model, 1024)
        super().__init__(dimension)

        # Use environment variable or passed key
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Voyage API key not found. Set VOYAGE_API_KEY environment variable or pass api_key."
            )

        # Initialize Voyage client
        self.client = voyageai.AsyncClient(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens

    def _truncate_text(self, text: str) -> str:
        """
        Truncate text if it exceeds token limit.
        Voyage doesn't provide a tokenizer, so we estimate based on characters.

        Args:
            text: Input text to potentially truncate

        Returns:
            Truncated text if needed
        """
        # Rough estimation: ~3.5 characters per token for code
        max_chars = int(self.max_tokens * 3.5)
        if len(text) > max_chars:
            logger.warning(
                f"Text exceeds estimated {self.max_tokens} tokens. Truncating for embedding."
            )
            return text[:max_chars]
        return text

    async def embed_text(
        self,
        text: str,
        input_type: Literal["document", "query"] = "document",
        native_format: bool = False,
        validate_dimensions: bool = False,
    ) -> list[float] | np.ndarray:
        """
        Generate embeddings for given text using Voyage AI API.

        Args:
            text: Input text to embed
            input_type: Type of input - "document" for indexing, "query" for search
            native_format: If True, return Neo4j-compatible numpy array
            validate_dimensions: If True, validate vector dimensions

        Returns:
            Embedding vector as a list of floats or numpy array
        """
        # Check for empty or whitespace-only text
        if not text or not text.strip():
            logger.info(
                f"Empty or whitespace-only text detected. Returning zero vector of dimension {self.dimension}"
            )
            zero_vector = [0.0] * self.dimension

            # Return native format if requested
            if native_format:
                return np.zeros(self.dimension, dtype=np.float32)

            return zero_vector

        # Truncate text if needed
        text = self._truncate_text(text)

        try:
            # Use Voyage embeddings API
            response = await self.client.embed(
                texts=[text], model=self.model, input_type=input_type
            )

            # Extract the first embedding (for single text input)
            embedding = response.embeddings[0]

            # Validate dimensions if requested
            if validate_dimensions and not self.validate_vector_dimensions(embedding):
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.dimension}, got {len(embedding)}"
                )

            # Return native format if requested
            if native_format:
                try:
                    neo4j_vector = self.to_neo4j_vector(embedding)
                    return self.normalize_vector(neo4j_vector)
                except Exception as e:
                    logger.warning(
                        f"Failed to convert to native format: {e}. Returning list format."
                    )
                    return embedding

            return embedding

        except Exception as e:
            logger.error(f"Voyage embedding generation failed: {e}")
            raise

    async def embed_batch(
        self,
        texts: list[str],
        input_type: Literal["document", "query"] = "document",
        native_format: bool = False,
        validate_dimensions: bool = False,
    ) -> list[list[float] | np.ndarray]:
        """
        Generate embeddings for multiple texts in batch.

        Args:
            texts: List of texts to embed
            input_type: Type of input - "document" for indexing, "query" for search
            native_format: If True, return Neo4j-compatible numpy arrays
            validate_dimensions: If True, validate vector dimensions

        Returns:
            List of embedding vectors
        """
        # Keep track of which texts are empty and their original positions
        non_empty_indices = []
        non_empty_texts = []

        for i, text in enumerate(texts):
            # Check if text is not empty and not just whitespace
            if text and text.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(self._truncate_text(text))

        # Handle case where all texts are empty
        if not non_empty_texts:
            logger.info(
                f"All {len(texts)} texts are empty or whitespace-only. Returning zero vectors."
            )
            if native_format:
                return [np.zeros(self.dimension, dtype=np.float32) for _ in texts]
            else:
                return [[0.0] * self.dimension for _ in texts]

        # Log if some texts were empty
        if len(non_empty_texts) < len(texts):
            empty_count = len(texts) - len(non_empty_texts)
            logger.info(
                f"Found {empty_count} empty/whitespace-only texts out of {len(texts)}. Will return zero vectors for empty texts."
            )

        try:
            # Use Voyage embeddings API for batch with only non-empty texts
            response = await self.client.embed(
                texts=non_empty_texts, model=self.model, input_type=input_type
            )

            embeddings = response.embeddings

            # Validate dimensions if requested
            if validate_dimensions:
                for i, embedding in enumerate(embeddings):
                    if not self.validate_vector_dimensions(embedding):
                        raise ValueError(
                            f"Embedding dimension mismatch at index {i}: "
                            f"expected {self.dimension}, got {len(embedding)}"
                        )

            # Reassemble results with zero vectors for empty texts
            result: list[list[float] | np.ndarray] = []
            embedding_idx = 0

            for i in range(len(texts)):
                if i in non_empty_indices:
                    # This text had content, use the generated embedding
                    embedding = embeddings[embedding_idx]
                    embedding_idx += 1

                    # Convert to native format if requested
                    if native_format:
                        try:
                            neo4j_vector = self.to_neo4j_vector(embedding)
                            result.append(self.normalize_vector(neo4j_vector))
                        except Exception as e:
                            logger.warning(
                                f"Failed to convert to native format: {e}. Using list format."
                            )
                            result.append(embedding)
                    else:
                        result.append(embedding)
                else:
                    # This text was empty, use zero vector
                    if native_format:
                        result.append(np.zeros(self.dimension, dtype=np.float32))
                    else:
                        result.append([0.0] * self.dimension)

            return result

        except Exception as e:
            logger.error(f"Voyage batch embedding generation failed: {e}")
            raise

    async def embed(self, text: str) -> list[float]:
        """
        Legacy method for backward compatibility.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as a list of floats
        """
        return await self.embed_text(text)
