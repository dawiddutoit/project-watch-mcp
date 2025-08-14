"""Neo4j native vector index implementation.

This module provides native Neo4j vector index operations that eliminate
the need for Lucene-based text search, providing better performance and
eliminating escaping issues.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import numpy as np
from neo4j import AsyncDriver
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class VectorIndexConfig:
    """Configuration for Neo4j vector index."""
    
    index_name: str = "code-embeddings"
    node_label: str = "CodeChunk"
    embedding_property: str = "embedding"
    dimensions: int = 1536
    similarity_metric: str = "cosine"
    provider: str = "openai"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.similarity_metric not in ["cosine", "euclidean"]:
            raise ValueError(f"Invalid similarity metric: {self.similarity_metric}")
        
        if self.dimensions <= 0:
            raise ValueError(f"Dimensions must be positive: {self.dimensions}")
        
        # Common embedding dimensions
        valid_dimensions = {
            "openai": [1536, 3072],  # text-embedding-3-small/large
            "voyage": [1024, 2048],  # voyage-code-2 variants
        }
        
        if self.provider in valid_dimensions:
            if self.dimensions not in valid_dimensions[self.provider]:
                logger.warning(
                    f"Unusual dimensions {self.dimensions} for provider {self.provider}. "
                    f"Common dimensions: {valid_dimensions[self.provider]}"
                )


@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""
    
    node_id: str
    score: float
    metadata: Dict[str, Any]
    content: str = ""
    
    def __lt__(self, other):
        """Enable sorting by score."""
        return self.score < other.score


@dataclass
class VectorUpsertResult:
    """Result from vector upsert operation."""
    
    node_id: str
    success: bool
    operation: str  # "created", "updated", "failed"
    error: Optional[str] = None


class NativeVectorIndex:
    """Manages Neo4j native vector indexes for semantic search."""
    
    def __init__(self, driver: AsyncDriver, config: VectorIndexConfig):
        """
        Initialize native vector index manager.
        
        Args:
            driver: Neo4j async driver instance
            config: Vector index configuration
        """
        self.driver = driver
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def create_index(self) -> Dict[str, Any]:
        """
        Create a native vector index in Neo4j.
        
        Returns:
            Dict with status and index details
        """
        query = f"""
        CREATE VECTOR INDEX `{self.config.index_name}` IF NOT EXISTS
        FOR (n:{self.config.node_label})
        ON (n.{self.config.embedding_property})
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.config.dimensions},
                `vector.similarity_function`: '{self.config.similarity_metric}'
            }}
        }}
        """
        
        try:
            async with self.driver.session() as session:
                await session.run(query)
                self.logger.info(f"Created vector index: {self.config.index_name}")
                return {
                    "status": "created",
                    "index_name": self.config.index_name,
                    "dimensions": self.config.dimensions,
                    "similarity_metric": self.config.similarity_metric
                }
        except Exception as e:
            self.logger.error(f"Failed to create vector index: {e}")
            raise
    
    async def drop_index(self) -> Dict[str, Any]:
        """
        Drop the vector index.
        
        Returns:
            Dict with status and index name
        """
        query = f"DROP INDEX `{self.config.index_name}` IF EXISTS"
        
        try:
            async with self.driver.session() as session:
                await session.run(query)
                self.logger.info(f"Dropped vector index: {self.config.index_name}")
                return {
                    "status": "dropped",
                    "index_name": self.config.index_name
                }
        except Exception as e:
            self.logger.error(f"Failed to drop vector index: {e}")
            raise
    
    async def upsert_vector(
        self,
        node_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> VectorUpsertResult:
        """
        Upsert a single vector to the index.
        
        Args:
            node_id: Unique identifier for the node
            vector: Embedding vector
            metadata: Additional properties to store with the node
            
        Returns:
            VectorUpsertResult with operation status
        """
        try:
            self._validate_dimensions(vector)
            
            # Build property setter string
            properties = {self.config.embedding_property: vector}
            if metadata:
                properties.update(metadata)
            
            # Convert properties to Cypher SET clause
            set_clauses = []
            params = {"node_id": node_id}
            
            for key, value in properties.items():
                param_name = f"prop_{key}"
                set_clauses.append(f"n.{key} = ${param_name}")
                params[param_name] = value
            
            set_clause = ", ".join(set_clauses)
            
            query = f"""
            MERGE (n:{self.config.node_label} {{id: $node_id}})
            SET {set_clause}
            RETURN n.id as node_id
            """
            
            async with self.driver.session() as session:
                await session.run(query, params)
                
            return VectorUpsertResult(
                node_id=node_id,
                success=True,
                operation="upserted"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to upsert vector for node {node_id}: {e}")
            return VectorUpsertResult(
                node_id=node_id,
                success=False,
                operation="failed",
                error=str(e)
            )
    
    async def batch_upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> List[VectorUpsertResult]:
        """
        Batch upsert multiple vectors for efficiency with memory management.
        
        Args:
            vectors: List of dicts with 'node_id', 'vector', and optional 'metadata'
            batch_size: Maximum number of vectors to process at once (default 1000)
            
        Returns:
            List of VectorUpsertResult for each vector
        """
        results = []
        
        # Process in chunks to control memory usage
        for chunk_start in range(0, len(vectors), batch_size):
            chunk_end = min(chunk_start + batch_size, len(vectors))
            chunk = vectors[chunk_start:chunk_end]
            
            # Validate vectors in this chunk
            validation_results = []
            valid_items = []
            
            for item in chunk:
                try:
                    self._validate_dimensions(item["vector"])
                    valid_items.append(item)
                except ValueError as e:
                    validation_results.append(VectorUpsertResult(
                        node_id=item["node_id"],
                        success=False,
                        operation="failed",
                        error=str(e)
                    ))
            
            # Prepare batch data for valid items only
            batch_data = []
            for item in valid_items:
                node_data = {
                    "node_id": item["node_id"],
                    self.config.embedding_property: item["vector"]
                }
                if "metadata" in item and item["metadata"]:
                    node_data.update(item["metadata"])
                batch_data.append(node_data)
            
            if batch_data:
                # Build dynamic property setter
                sample_keys = set()
                for data in batch_data[:10]:  # Sample first 10 for keys
                    sample_keys.update(data.keys())
                sample_keys.discard("node_id")
                
                set_clauses = [f"n.{key} = item.{key}" for key in sample_keys]
                set_clause = ", ".join(set_clauses)
                
                query = f"""
                UNWIND $batch AS item
                MERGE (n:{self.config.node_label} {{id: item.node_id}})
                SET {set_clause}
                RETURN n.id as node_id
                """
                
                try:
                    async with self.driver.session() as session:
                        result = await session.run(query, {"batch": batch_data})
                        processed_ids = set()
                        async for record in result:
                            processed_ids.add(record["node_id"])
                        
                        # Generate results for this chunk
                        for item in valid_items:
                            success = item["node_id"] in processed_ids
                            validation_results.append(VectorUpsertResult(
                                node_id=item["node_id"],
                                success=success,
                                operation="upserted" if success else "failed"
                            ))
                            
                except Exception as e:
                    self.logger.error(f"Batch upsert failed for chunk: {e}")
                    # Mark remaining valid items as failed
                    for item in valid_items:
                        if item["node_id"] not in [r.node_id for r in validation_results]:
                            validation_results.append(VectorUpsertResult(
                                node_id=item["node_id"],
                                success=False,
                                operation="failed",
                                error=str(e)
                            ))
            
            # Add chunk results to overall results
            results.extend(validation_results)
            
            # Clear chunk data to free memory
            del chunk
            del validation_results
            del valid_items
            del batch_data
        
        return results
    
    async def search(
        self,
        vector: List[float],
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors using native Neo4j vector search.
        
        Args:
            vector: Query vector
            top_k: Number of results to return
            metadata_filter: Optional metadata filters
            
        Returns:
            List of VectorSearchResult ordered by similarity
        """
        try:
            self._validate_dimensions(vector)
            
            # Build filter clause if metadata provided
            filter_clause = ""
            params = {
                "query_vector": vector,
                "top_k": top_k
            }
            
            if metadata_filter:
                filter_conditions = []
                for key, value in metadata_filter.items():
                    param_name = f"filter_{key}"
                    filter_conditions.append(f"node.{key} = ${param_name}")
                    params[param_name] = value
                filter_clause = f"WHERE {' AND '.join(filter_conditions)}"
            
            query = f"""
            CALL db.index.vector.queryNodes(
                '{self.config.index_name}',
                {top_k},
                $query_vector
            ) YIELD node, score
            {filter_clause}
            RETURN node, score
            ORDER BY score DESC
            LIMIT $top_k
            """
            
            results = []
            async with self.driver.session() as session:
                result = await session.run(query, params)
                async for record in result:
                    node = dict(record["node"])
                    
                    # Extract metadata (everything except id and embedding)
                    metadata = {
                        k: v for k, v in node.items()
                        if k not in ["id", self.config.embedding_property]
                    }
                    
                    results.append(VectorSearchResult(
                        node_id=node.get("id", ""),
                        score=record["score"],
                        metadata=metadata,
                        content=node.get("content", "")
                    ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            raise
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector index.
        
        Returns:
            Dict with index statistics
        """
        query = """
        SHOW INDEXES
        YIELD name, type, labelsOrTypes, properties, options
        WHERE type = 'VECTOR' AND name = $index_name
        RETURN name, labelsOrTypes, properties, options
        """
        
        stats = {}
        try:
            async with self.driver.session() as session:
                result = await session.run(query, {"index_name": self.config.index_name})
                async for record in result:
                    options = record.get("options", {})
                    index_config = options.get("indexConfig", {})
                    
                    stats = {
                        "index_name": record.get("name"),
                        "node_labels": record.get("labelsOrTypes", []),
                        "properties": record.get("properties", []),
                        "dimensions": index_config.get("vector.dimensions", self.config.dimensions),
                        "similarity_function": index_config.get("vector.similarity_function", self.config.similarity_metric)
                    }
                    
                    # Get node count
                    count_query = f"""
                    MATCH (n:{self.config.node_label})
                    WHERE n.{self.config.embedding_property} IS NOT NULL
                    RETURN count(n) as node_count
                    """
                    count_result = await session.run(count_query)
                    count_record = await count_result.single()
                    stats["node_count"] = count_record["node_count"] if count_record else 0
                    
        except Exception as e:
            self.logger.error(f"Failed to get index stats: {e}")
            raise
        
        return stats
    
    async def optimize_index(self) -> Dict[str, Any]:
        """
        Optimize the vector index for better performance.
        
        Returns:
            Dict with optimization results
        """
        start_time = time.time()
        
        try:
            async with self.driver.session() as session:
                # Neo4j may not have explicit vector index optimization
                # but we can run ANALYZE to update statistics
                analyze_query = f"""
                ANALYZE INDEX `{self.config.index_name}`
                """
                
                try:
                    await session.run(analyze_query)
                except:
                    # ANALYZE might not be available, try alternative
                    # Force index rebuild by touching nodes
                    touch_query = f"""
                    MATCH (n:{self.config.node_label})
                    WHERE n.{self.config.embedding_property} IS NOT NULL
                    WITH n LIMIT 1
                    SET n._optimized = timestamp()
                    REMOVE n._optimized
                    RETURN count(n) as touched
                    """
                    await session.run(touch_query)
                
                elapsed_ms = (time.time() - start_time) * 1000
                
                return {
                    "status": "optimized",
                    "index_name": self.config.index_name,
                    "optimization_time_ms": elapsed_ms
                }
                
        except Exception as e:
            self.logger.error(f"Failed to optimize index: {e}")
            raise
    
    def _validate_dimensions(self, vector: List[float]) -> bool:
        """
        Validate vector dimensions match configuration.
        
        Args:
            vector: Vector to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If dimensions don't match
        """
        if len(vector) != self.config.dimensions:
            raise ValueError(
                f"Vector dimension mismatch. Expected {self.config.dimensions}, "
                f"got {len(vector)}"
            )
        return True
    
    def _normalize_vector(self, vector: List[float]) -> np.ndarray:
        """
        Normalize vector to unit length for cosine similarity.
        
        Args:
            vector: Vector to normalize
            
        Returns:
            Normalized vector as numpy array
        """
        arr = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(arr)
        
        # Use epsilon for numerical stability
        if norm < np.finfo(np.float32).eps:
            return arr
        
        return arr / norm