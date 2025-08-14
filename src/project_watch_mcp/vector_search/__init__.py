"""Vector search module for Neo4j native vector capabilities.

This module provides native Neo4j vector index operations including:
- Index creation and management
- Vector upsert operations
- Similarity search
- Batch processing
"""

from .neo4j_native_vectors import (
    NativeVectorIndex,
    VectorSearchResult,
    VectorUpsertResult,
    VectorIndexConfig,
)

__all__ = [
    "NativeVectorIndex",
    "VectorSearchResult", 
    "VectorUpsertResult",
    "VectorIndexConfig",
]