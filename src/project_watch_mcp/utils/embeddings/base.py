"""Base class for embedding providers with native Neo4j vector support."""

from typing import List, Union
import numpy as np


class EmbeddingsProvider:
    """Base class for embedding providers with native Neo4j vector support."""

    def __init__(self, dimension: int = 1536):
        """Initialize with a default embedding dimension."""
        self.dimension = dimension

    async def embed_text(self, text: str) -> list[float]:
        """Generate embeddings for given text."""
        raise NotImplementedError("Subclasses must implement this method")
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def to_neo4j_vector(self, embedding: List[float]) -> np.ndarray:
        """
        Convert embedding list to Neo4j-compatible numpy array.
        
        Args:
            embedding: List of float values
            
        Returns:
            Numpy array with float32 dtype for Neo4j compatibility
        """
        return np.array(embedding, dtype=np.float32)
    
    def normalize_vector(self, vector: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Normalize vector for cosine similarity in Neo4j.
        
        Args:
            vector: Input vector as list or numpy array
            
        Returns:
            Normalized numpy array with magnitude 1
        """
        arr = np.array(vector, dtype=np.float32) if not isinstance(vector, np.ndarray) else vector
        
        # Handle zero vector case - check with epsilon for numerical stability
        norm = np.linalg.norm(arr)
        if norm < np.finfo(np.float32).eps:
            return np.zeros_like(arr, dtype=np.float32)
        
        return arr / norm
    
    def validate_vector_dimensions(self, vector: Union[List[float], np.ndarray]) -> bool:
        """
        Validate that vector has expected dimensions.
        
        Args:
            vector: Vector to validate
            
        Returns:
            True if dimensions match, False otherwise
        """
        length = len(vector) if isinstance(vector, list) else vector.shape[0]
        return length == self.dimension