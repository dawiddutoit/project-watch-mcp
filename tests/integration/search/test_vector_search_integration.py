"""Integration tests for Neo4j native vector search functionality.

Tests the complete integration with real Neo4j operations and embeddings.
"""

import pytest
import asyncio
from typing import List, Dict, Any
import os
from neo4j import AsyncGraphDatabase
import numpy as np

from project_watch_mcp.vector_search.neo4j_native_vectors import (
    NativeVectorIndex,
    VectorIndexConfig,
)
from project_watch_mcp.utils.embeddings.base import EmbeddingsProvider


@pytest.mark.integration
@pytest.mark.asyncio
class TestVectorSearchIntegration:
    """Integration tests with real Neo4j instance."""
    
    @pytest.fixture
    async def neo4j_driver(self):
        """Create real Neo4j driver for integration testing."""
        # Skip if no Neo4j is available
        neo4j_url = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        PROJECT_WATCH_USER = os.getenv("PROJECT_WATCH_USER", "neo4j")
        PROJECT_WATCH_PASSWORD = os.getenv("PROJECT_WATCH_PASSWORD", "password")
        
        try:
            driver = AsyncGraphDatabase.driver(
                neo4j_url,
                auth=(PROJECT_WATCH_USER, PROJECT_WATCH_PASSWORD)
            )
            # Test connection
            async with driver.session() as session:
                await session.run("RETURN 1")
            yield driver
            await driver.close()
        except Exception as e:
            pytest.skip(f"Neo4j not available: {e}")
    
    @pytest.fixture
    async def vector_index(self, neo4j_driver):
        """Create vector index instance for testing."""
        config = VectorIndexConfig(
            index_name="test-vector-index",
            node_label="TestChunk",
            dimensions=1536,
            similarity_metric="cosine"
        )
        index = NativeVectorIndex(neo4j_driver, config)
        
        # Clean up any existing index and nodes
        try:
            await index.drop_index()
        except:
            pass
        
        # Delete any existing test nodes
        async with neo4j_driver.session() as session:
            await session.run(f"MATCH (n:{config.node_label}) DELETE n")
        
        yield index
        
        # Cleanup after test
        try:
            await index.drop_index()
        except:
            pass
        
        # Delete all test nodes
        async with neo4j_driver.session() as session:
            await session.run(f"MATCH (n:{config.node_label}) DELETE n")
    
    async def test_complete_vector_workflow(self, vector_index):
        """Test complete workflow: create, upsert, search, drop."""
        # Create index
        result = await vector_index.create_index()
        assert result["status"] == "created"
        
        # Create sample vectors (simulating embeddings)
        sample_texts = [
            "Python is a high-level programming language",
            "JavaScript is used for web development",
            "Machine learning uses statistical methods",
            "Python is great for data science and ML",
            "Neo4j is a graph database"
        ]
        
        # Generate mock embeddings (in real scenario, use OpenAI/Voyage)
        vectors = []
        for i, text in enumerate(sample_texts):
            # Create somewhat meaningful mock embeddings
            base_vector = np.random.randn(1536) * 0.1
            # Add some pattern to make similar texts have similar vectors
            if "Python" in text:
                base_vector[0:100] += 0.5
            if "machine learning" in text.lower() or "ml" in text.lower():
                base_vector[100:200] += 0.5
            if "database" in text.lower():
                base_vector[200:300] += 0.5
            
            vectors.append({
                "node_id": f"chunk-{i}",
                "vector": base_vector.tolist(),
                "metadata": {
                    "text": text,
                    "index": i,
                    "language": "python" if "Python" in text else "other"
                }
            })
        
        # Batch upsert
        upsert_results = await vector_index.batch_upsert_vectors(vectors)
        assert len(upsert_results) == 5
        assert all(r.success for r in upsert_results)
        
        # Create query vector (similar to Python-related content)
        query_vector = np.random.randn(1536) * 0.1
        query_vector[0:100] += 0.5  # Python similarity
        query_vector[100:200] += 0.3  # Some ML similarity
        
        # Search without filter
        search_results = await vector_index.search(
            vector=query_vector.tolist(),
            top_k=3
        )
        
        assert len(search_results) <= 3
        # Results should be sorted by score
        if len(search_results) > 1:
            assert search_results[0].score >= search_results[1].score
        
        # Search with metadata filter
        filtered_results = await vector_index.search(
            vector=query_vector.tolist(),
            top_k=5,
            metadata_filter={"language": "python"}
        )
        
        # Should only return Python-related chunks
        for result in filtered_results:
            assert result.metadata.get("language") == "python"
        
        # Get index stats
        stats = await vector_index.get_index_stats()
        assert stats["node_count"] == 5
        assert stats["dimensions"] == 1536
        
        # Optimize index
        optimize_result = await vector_index.optimize_index()
        assert optimize_result["status"] == "optimized"
        
        # Drop index
        drop_result = await vector_index.drop_index()
        assert drop_result["status"] == "dropped"
    
    async def test_vector_search_performance(self, vector_index):
        """Test performance with larger dataset."""
        # Create index
        await vector_index.create_index()
        
        # Generate 100 vectors
        batch_size = 50
        total_vectors = 100
        
        for batch_start in range(0, total_vectors, batch_size):
            batch_vectors = []
            for i in range(batch_start, min(batch_start + batch_size, total_vectors)):
                vector = np.random.randn(1536) * 0.1
                batch_vectors.append({
                    "node_id": f"perf-chunk-{i}",
                    "vector": vector.tolist(),
                    "metadata": {
                        "batch": batch_start // batch_size,
                        "index": i
                    }
                })
            
            results = await vector_index.batch_upsert_vectors(batch_vectors)
            assert all(r.success for r in results)
        
        # Verify all inserted
        stats = await vector_index.get_index_stats()
        assert stats["node_count"] == total_vectors
        
        # Performance test: search should be fast
        import time
        query_vector = np.random.randn(1536).tolist()
        
        start_time = time.time()
        results = await vector_index.search(query_vector, top_k=10)
        search_time = time.time() - start_time
        
        assert len(results) == 10
        # Search should be reasonably fast (< 1 second for 100 vectors)
        assert search_time < 1.0
    
    async def test_dimension_mismatch_handling(self, vector_index):
        """Test handling of dimension mismatches."""
        await vector_index.create_index()
        
        # Try to insert vector with wrong dimensions
        wrong_vector = {
            "node_id": "wrong-dims",
            "vector": [0.1] * 100,  # Wrong size
            "metadata": {}
        }
        
        results = await vector_index.batch_upsert_vectors([wrong_vector])
        assert len(results) == 1
        assert not results[0].success
        assert "dimension mismatch" in results[0].error.lower()
    
    async def test_concurrent_operations(self, vector_index):
        """Test concurrent vector operations."""
        await vector_index.create_index()
        
        # Prepare vectors
        vectors = [
            {
                "node_id": f"concurrent-{i}",
                "vector": np.random.randn(1536).tolist(),
                "metadata": {"group": i % 3}
            }
            for i in range(30)
        ]
        
        # Split into batches for concurrent insertion
        batch1 = vectors[:10]
        batch2 = vectors[10:20]
        batch3 = vectors[20:]
        
        # Insert concurrently
        results = await asyncio.gather(
            vector_index.batch_upsert_vectors(batch1),
            vector_index.batch_upsert_vectors(batch2),
            vector_index.batch_upsert_vectors(batch3),
            return_exceptions=True
        )
        
        # Check all succeeded
        for batch_results in results:
            if isinstance(batch_results, Exception):
                raise batch_results
            assert all(r.success for r in batch_results)
        
        # Verify all inserted
        stats = await vector_index.get_index_stats()
        assert stats["node_count"] == 30
        
        # Concurrent searches
        search_vectors = [np.random.randn(1536).tolist() for _ in range(5)]
        search_tasks = [
            vector_index.search(vec, top_k=5)
            for vec in search_vectors
        ]
        
        search_results = await asyncio.gather(*search_tasks)
        
        # All searches should return results
        for results in search_results:
            assert len(results) <= 5
            # Results should be sorted by score
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score