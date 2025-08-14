"""Vector search support utilities for Neo4j integration.

This module provides utilities and helpers for working with Neo4j native vector indexes
and LangChain integration for semantic search capabilities.
"""

from typing import Dict, List, Optional, Any
import numpy as np
from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import VoyageEmbeddings
from langchain.schema import Document

__all__ = [
    "create_langchain_embeddings",
    "create_neo4j_vector_store",
    "convert_to_numpy_array",
    "cosine_similarity",
]


def create_langchain_embeddings(provider_type: str = "openai", **kwargs) -> Any:
    """
    Create LangChain-compatible embeddings provider.
    
    Args:
        provider_type: Type of embeddings provider ("openai" or "voyage")
        **kwargs: Provider-specific configuration
        
    Returns:
        LangChain embeddings instance
        
    Raises:
        ValueError: If provider type is unknown
    """
    if provider_type.lower() == "openai":
        # OpenAI configuration
        return OpenAIEmbeddings(
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model", "text-embedding-3-small"),
            dimensions=kwargs.get("dimensions", 1536),
            max_retries=kwargs.get("max_retries", 3),
        )
    elif provider_type.lower() == "voyage":
        # Voyage AI configuration
        return VoyageEmbeddings(
            voyage_api_key=kwargs.get("api_key"),
            model=kwargs.get("model", "voyage-code-2"),
            batch_size=kwargs.get("batch_size", 8),
        )
    else:
        raise ValueError(
            f"Unknown provider type: {provider_type}. "
            f"Supported: 'openai', 'voyage'"
        )


def create_neo4j_vector_store(
    url: str,
    username: str,
    password: str,
    embeddings: Any,
    index_name: str = "code-embeddings",
    node_label: str = "CodeChunk",
    **kwargs
) -> Neo4jVector:
    """
    Create a Neo4j vector store instance for semantic search.
    
    Args:
        url: Neo4j database URL
        username: Neo4j username
        password: Neo4j password
        embeddings: LangChain embeddings instance
        index_name: Name of the vector index in Neo4j
        node_label: Label of nodes to index
        **kwargs: Additional Neo4jVector configuration
        
    Returns:
        Configured Neo4jVector instance
    """
    return Neo4jVector(
        embedding=embeddings,
        url=url,
        username=username,
        password=password,
        index_name=index_name,
        node_label=node_label,
        text_node_property=kwargs.get("text_property", "content"),
        embedding_node_property=kwargs.get("embedding_property", "embedding"),
        search_type=kwargs.get("search_type", "hybrid"),
        keyword_index_name=kwargs.get("keyword_index", f"{index_name}-keywords"),
        retrieval_query=kwargs.get("retrieval_query"),
    )


def convert_to_numpy_array(embedding: List[float]) -> np.ndarray:
    """
    Convert embedding list to numpy array for vector operations.
    
    Args:
        embedding: List of float values
        
    Returns:
        Numpy array representation
    """
    return np.array(embedding, dtype=np.float32)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score between -1 and 1
    """
    arr1 = convert_to_numpy_array(vec1)
    arr2 = convert_to_numpy_array(vec2)
    
    dot_product = np.dot(arr1, arr2)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    
    # Avoid division by zero - use epsilon for numerical stability
    eps = np.finfo(np.float32).eps
    if norm1 < eps or norm2 < eps:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


class VectorIndexManager:
    """Manages Neo4j vector index creation and configuration."""
    
    def __init__(self, driver, dimensions: int = 1536):
        """
        Initialize vector index manager.
        
        Args:
            driver: Neo4j driver instance
            dimensions: Vector dimensions (default: 1536 for OpenAI)
        """
        self.driver = driver
        self.dimensions = dimensions
    
    async def create_vector_index(
        self,
        index_name: str = "code-embeddings",
        node_label: str = "CodeChunk",
        property_name: str = "embedding",
        similarity_function: str = "cosine"
    ) -> Dict[str, Any]:
        """
        Create a vector index in Neo4j.
        
        Args:
            index_name: Name for the index
            node_label: Label of nodes to index
            property_name: Property containing embeddings
            similarity_function: Similarity metric ('cosine' or 'euclidean')
            
        Returns:
            Index creation result
        """
        query = f"""
        CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS
        FOR (n:{node_label})
        ON (n.{property_name})
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.dimensions},
                `vector.similarity_function`: '{similarity_function}'
            }}
        }}
        """
        
        async with self.driver.session() as session:
            result = await session.run(query)
            return {"status": "created", "index_name": index_name}
    
    async def drop_vector_index(self, index_name: str) -> Dict[str, Any]:
        """
        Drop a vector index.
        
        Args:
            index_name: Name of the index to drop
            
        Returns:
            Drop operation result
        """
        query = f"DROP INDEX `{index_name}` IF EXISTS"
        
        async with self.driver.session() as session:
            result = await session.run(query)
            return {"status": "dropped", "index_name": index_name}
    
    async def list_vector_indexes(self) -> List[Dict[str, Any]]:
        """
        List all vector indexes in the database.
        
        Returns:
            List of vector index configurations
        """
        query = """
        SHOW INDEXES
        WHERE type = 'VECTOR'
        """
        
        async with self.driver.session() as session:
            result = await session.run(query)
            indexes = []
            async for record in result:
                indexes.append({
                    "name": record.get("name"),
                    "state": record.get("state"),
                    "labels": record.get("labelsOrTypes"),
                    "properties": record.get("properties"),
                    "options": record.get("options"),
                })
            return indexes