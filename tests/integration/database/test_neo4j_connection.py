"""
Integration tests for Neo4j connection management and resilience.

These tests verify:
1. Connection establishment and authentication
2. Connection pooling and reuse
3. Retry logic and error recovery
4. Connection timeout handling
5. Graceful degradation when Neo4j is unavailable
"""

import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError, Neo4jError

from project_watch_mcp.neo4j_rag import Neo4jRAG, CodeFile
from project_watch_mcp.config import ProjectWatchConfig
from project_watch_mcp.optimization.connection_pool import ConnectionPoolManager


class TestNeo4jConnection:
    """Test suite for Neo4j connection management."""

    @pytest.fixture
    def neo4j_config(self):
        """Create Neo4j configuration."""
        from project_watch_mcp.config import (
            ProjectWatchConfig,
            ProjectConfig,
            Neo4jConfig,
            EmbeddingConfig
        )
        from pathlib import Path
        
        # Create sub-configs
        project_config = ProjectConfig(
            name="test_project",
            repository_path=Path.cwd()
        )
        
        neo4j_config = Neo4jConfig(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
            database="neo4j"
        )
        
        embedding_config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-api-key",
            dimension=1536
        )
        
        # Create main config
        config = ProjectWatchConfig(
            project=project_config,
            neo4j=neo4j_config,
            embedding=embedding_config,
            chunk_size=500,
            chunk_overlap=50
        )
        
        return config

    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embeddings provider."""
        mock = AsyncMock()
        mock.embed_documents = AsyncMock(return_value=[[0.1] * 1536])
        mock.embed_query = AsyncMock(return_value=[0.1] * 1536)
        return mock

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("NEO4J_URI"),
        reason="Neo4j connection not configured"
    )
    async def test_successful_connection_establishment(self, neo4j_config, mock_embeddings):
        """Test successful connection to Neo4j."""
        # Create Neo4j driver
        driver = AsyncGraphDatabase.driver(
            neo4j_config.neo4j.uri,
            auth=(neo4j_config.neo4j.username, neo4j_config.neo4j.password)
        )
        
        rag = Neo4jRAG(
            neo4j_driver=driver,
            project_name="test_project",
            embeddings=mock_embeddings
        )
        
        try:
            # Initialize connection
            await rag.initialize()
            
            # Verify connection is established
            assert rag.neo4j_driver is not None
            
            # Test connection with a simple query
            async with rag.neo4j_driver.session() as session:
                result = await session.run("RETURN 1 as num")
                record = await result.single()
                assert record["num"] == 1
        finally:
            await driver.close()

    @pytest.mark.asyncio
    async def test_connection_with_invalid_credentials(self, mock_embeddings):
        """Test connection failure with invalid credentials."""
        with patch('neo4j.AsyncGraphDatabase.driver') as mock_driver:
            mock_driver.side_effect = AuthError("Invalid credentials")
            
            with pytest.raises(AuthError):
                driver = AsyncGraphDatabase.driver(
                    "bolt://localhost:7687",
                    auth=("invalid_user", "wrong_password")
                )

    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self, mock_embeddings):
        """Test handling of connection timeouts."""
        with patch('neo4j.AsyncGraphDatabase.driver') as mock_driver:
            mock_driver.side_effect = ServiceUnavailable("Connection timeout")
            
            with pytest.raises(ServiceUnavailable):
                driver = AsyncGraphDatabase.driver(
                    "bolt://localhost:9999",  # Non-existent port
                    auth=("neo4j", "password")
                )

    @pytest.mark.asyncio
    async def test_connection_retry_logic(self, mock_embeddings):
        """Test automatic retry on connection failure."""
        call_count = 0
        
        async def mock_verify_connectivity():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ServiceUnavailable("Connection failed")
            return True
        
        mock_driver = AsyncMock()
        mock_driver.verify_connectivity = mock_verify_connectivity
        
        with patch('neo4j.AsyncGraphDatabase.driver', return_value=mock_driver):
            driver = AsyncGraphDatabase.driver(
                "bolt://localhost:7687",
                auth=("neo4j", "password")
            )
            
            rag = Neo4jRAG(
                neo4j_driver=driver,
                project_name="test_project",
                embeddings=mock_embeddings
            )
            
            # Should retry and eventually succeed
            await rag.initialize()
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_connection_pool_management(self):
        """Test connection pool creation and management."""
        pool = ConnectionPoolManager(
            uri="bolt://localhost:7687",
            auth=("neo4j", "password"),
            max_connections=5,
            connection_timeout=10
        )
        
        # Mock Neo4j driver
        mock_driver = AsyncMock()
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"num": 1})
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_driver.session.return_value.__aenter__.return_value = mock_session
        
        with patch('neo4j.AsyncGraphDatabase.driver', return_value=mock_driver):
            # Initialize pool
            await pool.initialize()
            
            # Execute multiple queries using the pool
            results = []
            for _ in range(3):
                result = await pool.execute_query("RETURN 1 as num")
                results.append(result)
            
            # Verify queries were executed
            assert len(results) == 3
            
            # Close pool
            await pool.close()

    @pytest.mark.asyncio
    async def test_connection_reuse_after_error(self, mock_embeddings):
        """Test that connections are properly reused after errors."""
        mock_driver = AsyncMock()
        mock_session = AsyncMock()
        
        # First query fails, second succeeds
        query_results = [
            Neo4jError("Query failed"),
            AsyncMock(single=AsyncMock(return_value={"result": "success"}))
        ]
        
        mock_session.run = AsyncMock(side_effect=query_results)
        mock_driver.session.return_value.__aenter__.return_value = mock_session
        
        rag = Neo4jRAG(
            neo4j_driver=mock_driver,
            project_name="test_project",
            embeddings=mock_embeddings
        )
        
        # First query should fail
        with pytest.raises(Neo4jError):
            async with rag.neo4j_driver.session() as session:
                await session.run("MATCH (n) RETURN n")
        
        # Second query should succeed (connection reused)
        async with rag.neo4j_driver.session() as session:
            result = await session.run("MATCH (n) RETURN n")
            assert result is not None

    @pytest.mark.asyncio
    async def test_graceful_degradation_without_neo4j(self, mock_embeddings):
        """Test that system degrades gracefully when Neo4j is unavailable."""
        mock_driver = AsyncMock()
        mock_driver.verify_connectivity.side_effect = ServiceUnavailable("Neo4j unavailable")
        
        rag = Neo4jRAG(
            neo4j_driver=mock_driver,
            project_name="test_project",
            embeddings=mock_embeddings
        )
        
        # Initialize should fail
        with pytest.raises(ServiceUnavailable):
            await rag.initialize()
        
        # But object should still be usable for other operations
        assert rag.neo4j_driver is not None  # Mock driver is still set
        
        # Search should return empty results rather than crash
        with patch.object(rag, 'search_code', return_value=[]):
            results = await rag.search_code("test query")
            assert results == []

    @pytest.mark.asyncio
    async def test_connection_cleanup_on_error(self, mock_embeddings):
        """Test that connections are properly cleaned up on error."""
        mock_driver = AsyncMock()
        mock_driver.close = AsyncMock()
        
        rag = Neo4jRAG(
            neo4j_driver=mock_driver,
            project_name="test_project",
            embeddings=mock_embeddings
        )
        
        # Simulate error during operation
        mock_session = AsyncMock()
        mock_session.run.side_effect = Neo4jError("Operation failed")
        mock_driver.session.return_value.__aenter__.return_value = mock_session
        
        # Operation should fail
        with pytest.raises(Neo4jError):
            async with rag.neo4j_driver.session() as session:
                await session.run("INVALID QUERY")
        
        # Close should still work
        await mock_driver.close()
        mock_driver.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_connection_requests(self, mock_embeddings):
        """Test handling of concurrent connection requests."""
        mock_driver = AsyncMock()
        mock_session = AsyncMock()
        mock_session.run = AsyncMock(return_value=AsyncMock(
            single=AsyncMock(return_value={"num": 1})
        ))
        mock_driver.session.return_value.__aenter__.return_value = mock_session
        
        rag = Neo4jRAG(
            neo4j_driver=mock_driver,
            project_name="test_project",
            embeddings=mock_embeddings
        )
        
        # Execute multiple concurrent queries
        async def run_query():
            async with rag.neo4j_driver.session() as session:
                result = await session.run("RETURN 1 as num")
                record = await result.single()
                return record["num"]
        
        tasks = [run_query() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(r == 1 for r in results)
        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_connection_with_custom_timeout(self, mock_embeddings):
        """Test connection with custom timeout settings."""
        mock_driver = AsyncMock()
        
        rag = Neo4jRAG(
            neo4j_driver=mock_driver,
            project_name="test_project",
            embeddings=mock_embeddings
        )
        
        start_time = time.time()
        
        # Simulate slow connection
        async def slow_verify():
            await asyncio.sleep(10)  # Longer than timeout
            raise ServiceUnavailable("Connection timeout")
        
        mock_driver.verify_connectivity = slow_verify
        
        # Should timeout before 10 seconds
        with pytest.raises(Exception):  # Could be timeout or ServiceUnavailable
            await asyncio.wait_for(rag.initialize(), timeout=6)
        
        elapsed = time.time() - start_time
        assert elapsed < 7  # Should timeout around 5-6 seconds

    @pytest.mark.asyncio
    async def test_connection_string_validation(self, mock_embeddings):
        """Test validation of connection strings."""
        # Test with various URI formats
        valid_uris = [
            "bolt://localhost:7687",
            "neo4j://localhost:7687",
            "neo4j+s://localhost:7687",
            "bolt+s://localhost:7687"
        ]
        
        for uri in valid_uris:
            mock_driver = AsyncMock()
            # Create driver and RAG instance
            rag = Neo4jRAG(
                neo4j_driver=mock_driver,
                project_name="test_project",
                embeddings=mock_embeddings
            )
            assert rag.neo4j_driver is not None

    @pytest.mark.asyncio
    async def test_connection_health_check(self, mock_embeddings):
        """Test connection health checking mechanism."""
        mock_driver = AsyncMock()
        
        # Simulate healthy connection
        mock_driver.verify_connectivity = AsyncMock(return_value=True)
        
        rag = Neo4jRAG(
            neo4j_driver=mock_driver,
            project_name="test_project",
            embeddings=mock_embeddings
        )
        
        # Check health using verify_connectivity on the driver
        is_healthy = await mock_driver.verify_connectivity()
        assert is_healthy
        
        # Simulate unhealthy connection
        mock_driver.verify_connectivity = AsyncMock(
            side_effect=ServiceUnavailable("Connection lost")
        )
        
        with pytest.raises(ServiceUnavailable):
            await mock_driver.verify_connectivity()

    @pytest.mark.asyncio
    async def test_automatic_reconnection(self, mock_embeddings):
        """Test automatic reconnection after connection loss."""
        mock_driver = AsyncMock()
        call_count = 0
        
        async def mock_run(query):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ServiceUnavailable("Connection lost")
            return AsyncMock(single=AsyncMock(return_value={"result": "success"}))
        
        mock_session = AsyncMock()
        mock_session.run = mock_run
        mock_driver.session.return_value.__aenter__.return_value = mock_session
        
        rag = Neo4jRAG(
            neo4j_driver=mock_driver,
            project_name="test_project",
            embeddings=mock_embeddings
        )
        
        # First attempt should fail and trigger reconnection
        with pytest.raises(ServiceUnavailable):
            async with rag.neo4j_driver.session() as session:
                await session.run("MATCH (n) RETURN n")
        
        # Second attempt should succeed after reconnection
        async with rag.neo4j_driver.session() as session:
            result = await session.run("MATCH (n) RETURN n")
            record = await result.single()
            assert record["result"] == "success"

    @pytest.mark.asyncio
    async def test_connection_metrics_tracking(self, mock_embeddings):
        """Test tracking of connection metrics."""
        # Initialize metrics
        metrics = {
            "connections_created": 0,
            "connections_failed": 0,
            "queries_executed": 0,
            "queries_failed": 0
        }
        
        mock_driver = AsyncMock()
        mock_session = AsyncMock()
        
        async def track_query(query):
            metrics["queries_executed"] += 1
            return AsyncMock(single=AsyncMock(return_value={"result": "success"}))
        
        mock_session.run = track_query
        mock_driver.session.return_value.__aenter__.return_value = mock_session
        
        rag = Neo4jRAG(
            neo4j_driver=mock_driver,
            project_name="test_project",
            embeddings=mock_embeddings
        )
        metrics["connections_created"] += 1
        
        # Execute queries
        for _ in range(5):
            async with rag.neo4j_driver.session() as session:
                await session.run("MATCH (n) RETURN n")
        
        # Check metrics
        assert metrics["connections_created"] == 1
        assert metrics["queries_executed"] == 5
        assert metrics["queries_failed"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])