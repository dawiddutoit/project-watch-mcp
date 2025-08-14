"""OpenAI embeddings provider with native Neo4j vector support."""

import logging
import os
from typing import List, Union

import numpy as np
import tiktoken
from openai import AsyncOpenAI

from .base import EmbeddingsProvider

logger = logging.getLogger(__name__)


class OpenAIEmbeddingsProvider(EmbeddingsProvider):
    """OpenAI embeddings provider using text-embedding-3-small model."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        max_tokens: int = 8191,
    ):
        """
        Initialize OpenAI Embeddings provider.

        Args:
            api_key: OpenAI API key (defaults to environment variable)
            model: OpenAI embedding model to use
            dimension: Embedding vector dimension
            max_tokens: Maximum tokens allowed by the model
        """
        super().__init__(dimension)

        # Use environment variable or passed key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set OPENAI_API_KEY environment variable or pass api_key."
            )

        # Initialize OpenAI client with new API
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens
        
        # Handle custom models that tiktoken doesn't recognize
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fall back to cl100k_base for unknown models
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    async def embed_text(
        self, 
        text: str, 
        native_format: bool = False,
        validate_dimensions: bool = False
    ) -> Union[list[float], np.ndarray]:
        """
        Generate embeddings for given text using OpenAI API.

        Args:
            text: Input text to embed
            native_format: If True, return Neo4j-compatible numpy array
            validate_dimensions: If True, validate vector dimensions

        Returns:
            Embedding vector as a list of floats or numpy array
        """
        # Truncate text if it exceeds token limit
        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.max_tokens:
            logger.warning(f"Text exceeds {self.max_tokens} tokens. Truncating for embedding.")
            tokens = tokens[: self.max_tokens]
            text = self.tokenizer.decode(tokens)

        try:
            # Use OpenAI embeddings API (new client)
            response = await self.client.embeddings.create(input=text, model=self.model)

            # Extract the first embedding (for single text input)
            embedding = response.data[0].embedding
            
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
            logger.error(f"OpenAI embedding generation failed: {e}")
            raise
    
    async def embed_batch(
        self,
        texts: List[str],
        native_format: bool = False,
        validate_dimensions: bool = False
    ) -> List[Union[List[float], np.ndarray]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            native_format: If True, return Neo4j-compatible numpy arrays
            validate_dimensions: If True, validate vector dimensions
            
        Returns:
            List of embedding vectors
        """
        # Truncate texts if needed
        truncated_texts = []
        for text in texts:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > self.max_tokens:
                logger.warning(f"Text exceeds {self.max_tokens} tokens. Truncating for embedding.")
                tokens = tokens[: self.max_tokens]
                text = self.tokenizer.decode(tokens)
            truncated_texts.append(text)
        
        try:
            # Use OpenAI embeddings API for batch
            response = await self.client.embeddings.create(
                input=truncated_texts,
                model=self.model
            )
            
            # Extract embeddings
            embeddings = [data.embedding for data in response.data]
            
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
            logger.error(f"OpenAI batch embedding generation failed: {e}")
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