"""Voyage AI embeddings provider optimized for code search with native Neo4j vector support."""

import logging
import os
from typing import Literal, List, Union

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
                "Voyage API key not found. "
                "Set VOYAGE_API_KEY environment variable or pass api_key."
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
        validate_dimensions: bool = False
    ) -> Union[list[float], np.ndarray]:
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
                    f"Embedding dimension mismatch: expected {self.dimension}, "
                    f"got {len(embedding)}"
                )
            
            # Return native format if requested
            if native_format:
                try:
                    neo4j_vector = self.to_neo4j_vector(embedding)
                    return self.normalize_vector(neo4j_vector)
                except Exception as e:
                    logger.warning(f"Failed to convert to native format: {e}. Returning list format.")
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
        validate_dimensions: bool = False
    ) -> List[Union[List[float], np.ndarray]]:
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
        # Truncate texts if needed
        truncated_texts = [self._truncate_text(text) for text in texts]

        try:
            # Use Voyage embeddings API for batch
            response = await self.client.embed(
                texts=truncated_texts, model=self.model, input_type=input_type
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
            
            # Convert to native format if requested
            if native_format:
                native_embeddings = []
                for embedding in embeddings:
                    try:
                        neo4j_vector = self.to_neo4j_vector(embedding)
                        native_embeddings.append(self.normalize_vector(neo4j_vector))
                    except Exception as e:
                        logger.warning(f"Failed to convert to native format: {e}. Using list format.")
                        native_embeddings.append(embedding)
                return native_embeddings
            
            return embeddings

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