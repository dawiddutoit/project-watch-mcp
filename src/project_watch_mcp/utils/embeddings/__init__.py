"""Embedding providers for semantic code search."""

import logging
from typing import Optional

from .base import EmbeddingsProvider
from .openai import OpenAIEmbeddingsProvider
from .voyage import VoyageEmbeddingsProvider
from .vector_support import (
    VectorIndexManager,
    create_langchain_embeddings,
    create_neo4j_vector_store,
    convert_to_numpy_array,
    cosine_similarity,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EmbeddingsProvider",
    "OpenAIEmbeddingsProvider",
    "VoyageEmbeddingsProvider",
    "create_embeddings_provider",
    # Vector support exports
    "VectorIndexManager",
    "create_langchain_embeddings",
    "create_neo4j_vector_store",
    "convert_to_numpy_array",
    "cosine_similarity",
]


def create_embeddings_provider(provider_type: str = "openai", **kwargs) -> Optional[EmbeddingsProvider]:
    """
    Factory function to create embeddings provider instances.

    Args:
        provider_type: Type of provider ("openai", "voyage")
        **kwargs: Additional arguments passed to provider constructor

    Returns:
        Configured embeddings provider instance, or None if API key is missing

    Raises:
        ValueError: If provider type is unknown
    """
    providers = {
        "openai": OpenAIEmbeddingsProvider,
        "voyage": VoyageEmbeddingsProvider,
    }

    provider_class = providers.get(provider_type.lower())
    if not provider_class:
        raise ValueError(
            f"Unknown provider type: {provider_type}. " f"Available: {', '.join(providers.keys())}"
        )

    try:
        # Filter kwargs based on provider type
        if provider_type.lower() == "voyage":
            # VoyageEmbeddingsProvider accepts: api_key, model, max_tokens
            filtered_kwargs = {k: v for k, v in kwargs.items() 
                             if k in ["api_key", "model", "max_tokens"]}
        else:
            # OpenAIEmbeddingsProvider accepts: api_key, model, dimension, max_tokens
            filtered_kwargs = {k: v for k, v in kwargs.items() 
                             if k in ["api_key", "model", "dimension", "max_tokens"]}
            
        return provider_class(**filtered_kwargs)
    except ValueError as e:
        # If API key is missing, log warning and return None
        logger.warning(f"Failed to create {provider_type} provider: {e}")
        logger.warning("Embeddings disabled - no API key configured. Set OPENAI_API_KEY or VOYAGE_API_KEY environment variable.")
        return None