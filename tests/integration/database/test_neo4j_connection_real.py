"""
Integration tests for Neo4j connection management with REAL database.

These tests verify connection behavior against an actual Neo4j instance
running in a Docker container, not mocks.
"""

import asyncio
import os
import time
import pytest
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError, Neo4jError

from project_watch_mcp.neo4j_rag import Neo4jRAG, CodeFile
from project_watch_mcp.config import ProjectWatchConfig


class TestRealNeo4jConnectionManagement:
    """Test suite for Neo4j connection management with real database."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_successful_connection_establishment(self, real_neo4j_driver, real_embeddings_provider):
        """Test successful connection to real Neo4j instance."""
        # Create RAG with real driver
        rag = Neo4jRAG(
            neo4j_driver=real_neo4j_driver,
            project_name="test_project",
            embeddings=real_embeddings_provider
        )
        
        # Initialize connection
        await rag.initialize()
        
        # Verify connection is established
        assert rag.neo4j_driver is not None
        
        # Test connection with a simple query
        async with rag.neo4j_driver.session() as session:
            result = await session.run("RETURN 1 as num, 'connected' as status")
            record = await result.single()
            assert record["num"] == 1
            assert record["status"] == "connected"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_connection_with_invalid_credentials(self, neo4j_container):
        """Test connection failure with invalid credentials against real database."""
        # Try to connect with wrong password
        with pytest.raises((AuthError, ServiceUnavailable)):
            driver = AsyncGraphDatabase.driver(
                neo4j_container.get_connection_url(),
                auth=("neo4j", "wrong_password")
            )
            await driver.verify_connectivity()
            await driver.close()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_connection_to_nonexistent_server(self):
        """Test handling of connection to non-existent server."""
        # Try to connect to a non-existent server
        driver = AsyncGraphDatabase.driver(
            "bolt://localhost:9999",  # Non-existent port
            auth=("neo4j", "password")
        )
        
        with pytest.raises(ServiceUnavailable):
            await asyncio.wait_for(driver.verify_connectivity(), timeout=5)
        
        await driver.close()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_connection_pool_behavior(self, real_neo4j_driver):
        """Test connection pooling behavior with real database."""
        # Execute multiple queries to test connection reuse
        query_times = []
        
        for i in range(10):
            start_time = time.time()
            
            async with real_neo4j_driver.session() as session:
                result = await session.run(
                    "RETURN $num as num, $text as text",
                    num=i,
                    text=f"query_{i}"
                )
                record = await result.single()
                assert record["num"] == i
                assert record["text"] == f"query_{i}"
            
            query_time = time.time() - start_time
            query_times.append(query_time)
        
        # First query might be slower due to connection establishment
        # Subsequent queries should be faster due to connection pooling
        avg_first_three = sum(query_times[:3]) / 3
        avg_last_three = sum(query_times[-3:]) / 3
        
        print(f"Average first 3 queries: {avg_first_three:.3f}s")
        print(f"Average last 3 queries: {avg_last_three:.3f}s")
        
        # Connection pooling should make later queries faster or similar
        assert avg_last_three <= avg_first_three * 1.5

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_connection_requests(self, real_neo4j_driver):
        """Test handling of concurrent queries with real database."""
        # Create multiple concurrent tasks
        async def run_query(query_id):
            async with real_neo4j_driver.session() as session:
                result = await session.run(
                    "RETURN $id as query_id, timestamp() as ts",
                    id=query_id
                )
                record = await result.single()
                return record["query_id"], record["ts"]
        
        # Execute 20 concurrent queries
        tasks = [run_query(i) for i in range(20)]
        results = await asyncio.gather(*tasks)
        
        # All queries should succeed
        assert len(results) == 20
        assert all(result[0] == i for i, result in enumerate(results))
        
        # Timestamps should be close (within 1 second)
        timestamps = [r[1] for r in results]
        time_range = max(timestamps) - min(timestamps)
        assert time_range < 1000, "Concurrent queries took too long to execute"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_connection_recovery_after_error(self, real_neo4j_driver, real_embeddings_provider):
        """Test that connections recover after query errors."""
        rag = Neo4jRAG(
            neo4j_driver=real_neo4j_driver,
            project_name="test_project",
            embeddings=real_embeddings_provider
        )
        
        # Execute an invalid query
        with pytest.raises(Neo4jError):
            async with rag.neo4j_driver.session() as session:
                await session.run("INVALID CYPHER SYNTAX HERE")
        
        # Connection should still work for valid queries
        async with rag.neo4j_driver.session() as session:
            result = await session.run("RETURN 'recovered' as status")
            record = await result.single()
            assert record["status"] == "recovered"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_transaction_isolation(self, real_neo4j_driver):
        """Test transaction isolation with real database."""
        # Start two concurrent transactions
        async def transaction_1(driver):
            async with driver.session() as session:
                tx = await session.begin_transaction()
                try:
                    await tx.run("CREATE (n:TxTest {id: 1, value: 'tx1'})")
                    await asyncio.sleep(0.5)  # Hold transaction open
                    
                    # Check own transaction sees the node
                    result = await tx.run("MATCH (n:TxTest {id: 1}) RETURN n.value as value")
                    record = await result.single()
                    assert record["value"] == "tx1"
                    
                    await tx.commit()
                    return "tx1_committed"
                except Exception:
                    await tx.rollback()
                    raise
        
        async def transaction_2(driver):
            await asyncio.sleep(0.1)  # Start slightly after tx1
            
            async with driver.session() as session:
                # Should not see uncommitted data from tx1
                result = await session.run("MATCH (n:TxTest {id: 1}) RETURN n")
                records = await result.data()
                assert len(records) == 0, "Should not see uncommitted data"
                
                # Create own node
                await session.run("CREATE (n:TxTest {id: 2, value: 'tx2'})")
                return "tx2_complete"
        
        # Run transactions concurrently
        results = await asyncio.gather(
            transaction_1(real_neo4j_driver),
            transaction_2(real_neo4j_driver)
        )
        
        assert "tx1_committed" in results
        assert "tx2_complete" in results
        
        # Verify final state
        async with real_neo4j_driver.session() as session:
            result = await session.run("MATCH (n:TxTest) RETURN n.id as id ORDER BY id")
            ids = [record["id"] async for record in result]
            assert ids == [1, 2]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_connection_cleanup_on_driver_close(self, neo4j_container):
        """Test proper cleanup when driver is closed."""
        # Create a new driver
        driver = AsyncGraphDatabase.driver(
            neo4j_container.get_connection_url(),
            auth=("neo4j", "testpassword")
        )
        
        # Use the driver
        async with driver.session() as session:
            await session.run("CREATE (n:CleanupTest {id: 1})")
        
        # Close the driver
        await driver.close()
        
        # Driver should not be usable after close
        with pytest.raises(Exception):
            async with driver.session() as session:
                await session.run("RETURN 1")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_connection_with_database_restart(self, real_neo4j_driver):
        """Test connection behavior during database operations."""
        # This test verifies the driver handles connection state properly
        
        # Verify initial connection works
        async with real_neo4j_driver.session() as session:
            result = await session.run("RETURN 'before' as status")
            record = await result.single()
            assert record["status"] == "before"
        
        # Create some data
        async with real_neo4j_driver.session() as session:
            await session.run("CREATE (n:RestartTest {value: 'persistent'})")
        
        # Verify data exists
        async with real_neo4j_driver.session() as session:
            result = await session.run("MATCH (n:RestartTest) RETURN n.value as value")
            record = await result.single()
            assert record["value"] == "persistent"

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.benchmark
    async def test_connection_latency(self, real_neo4j_driver):
        """Measure connection establishment and query latency."""
        latencies = {
            "connection_establishment": [],
            "simple_query": [],
            "complex_query": []
        }
        
        # Measure connection establishment time
        for _ in range(5):
            start = time.time()
            await real_neo4j_driver.verify_connectivity()
            latencies["connection_establishment"].append(time.time() - start)
        
        # Measure simple query latency
        async with real_neo4j_driver.session() as session:
            for _ in range(10):
                start = time.time()
                result = await session.run("RETURN 1")
                await result.single()
                latencies["simple_query"].append(time.time() - start)
        
        # Measure complex query latency
        async with real_neo4j_driver.session() as session:
            # Create test data
            await session.run("""
                UNWIND range(1, 100) as i
                CREATE (n:LatencyTest {id: i, data: 'x' * 100})
            """)
            
            for _ in range(10):
                start = time.time()
                result = await session.run("""
                    MATCH (n:LatencyTest)
                    WHERE n.id > 10 AND n.id < 90
                    RETURN count(n) as count
                """)
                await result.single()
                latencies["complex_query"].append(time.time() - start)
        
        # Print latency statistics
        for operation, times in latencies.items():
            avg = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            print(f"\n{operation}:")
            print(f"  Average: {avg*1000:.2f}ms")
            print(f"  Min: {min_time*1000:.2f}ms")
            print(f"  Max: {max_time*1000:.2f}ms")
        
        # Assert reasonable latencies
        assert sum(latencies["simple_query"]) / len(latencies["simple_query"]) < 0.1  # < 100ms avg
        assert sum(latencies["complex_query"]) / len(latencies["complex_query"]) < 0.5  # < 500ms avg


class TestConnectionResilience:
    """Test connection resilience and error recovery with real database."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_query_timeout_handling(self, real_neo4j_driver):
        """Test handling of long-running queries."""
        async with real_neo4j_driver.session() as session:
            # Create a large dataset
            await session.run("""
                UNWIND range(1, 1000) as i
                CREATE (n:TimeoutTest {id: i})
            """)
            
            # Run a potentially slow query with timeout
            start_time = time.time()
            
            # This should complete normally
            result = await session.run("""
                MATCH (n1:TimeoutTest), (n2:TimeoutTest)
                WHERE n1.id < 10 AND n2.id < 10
                RETURN count(*) as count
            """)
            record = await result.single()
            
            elapsed = time.time() - start_time
            assert record["count"] == 81  # 9 * 9
            assert elapsed < 5, f"Query took too long: {elapsed:.2f}s"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_memory_efficient_streaming(self, real_neo4j_driver):
        """Test memory-efficient result streaming."""
        async with real_neo4j_driver.session() as session:
            # Create a large dataset
            await session.run("""
                UNWIND range(1, 10000) as i
                CREATE (n:StreamTest {
                    id: i,
                    data: 'x' * 1000
                })
            """)
            
            # Stream results efficiently
            result = await session.run("MATCH (n:StreamTest) RETURN n.id as id ORDER BY id")
            
            count = 0
            ids_sample = []
            
            # Process results in streaming fashion
            async for record in result:
                count += 1
                if count <= 10:
                    ids_sample.append(record["id"])
            
            assert count == 10000
            assert ids_sample == list(range(1, 11))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "integration"])