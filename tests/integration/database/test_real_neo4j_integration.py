"""
Real Neo4j database integration tests.

These tests use a real Neo4j database instance running in a Docker container
via testcontainers to ensure we're testing actual database behavior, not mocks.
"""

import asyncio
import pytest
from neo4j.exceptions import ServiceUnavailable, AuthError, Neo4jError

from project_watch_mcp.neo4j_rag import Neo4jRAG, CodeFile


class TestRealNeo4jConnection:
    """Test suite for real Neo4j database operations."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_connection_establishment(self, real_neo4j_driver):
        """Test establishing a real connection to Neo4j database."""
        # Verify connectivity
        await real_neo4j_driver.verify_connectivity()
        
        # Execute a simple query to ensure connection works
        async with real_neo4j_driver.session() as session:
            result = await session.run("RETURN 1 as number, 'hello' as greeting")
            record = await result.single()
            
            assert record["number"] == 1
            assert record["greeting"] == "hello"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_create_and_query_nodes(self, real_neo4j_driver):
        """Test creating and querying nodes in real database."""
        async with real_neo4j_driver.session() as session:
            # Create test nodes
            await session.run("""
                CREATE (f1:CodeFile {path: '/src/main.py', content: 'def main(): pass'})
                CREATE (f2:CodeFile {path: '/src/utils.py', content: 'def helper(): return 42'})
                CREATE (f3:CodeFile {path: '/tests/test_main.py', content: 'def test_main(): assert True'})
            """)
            
            # Query nodes
            result = await session.run("MATCH (f:CodeFile) RETURN f.path as path ORDER BY path")
            paths = [record["path"] async for record in result]
            
            assert len(paths) == 3
            assert paths == ['/src/main.py', '/src/utils.py', '/tests/test_main.py']
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_index_creation_and_usage(self, real_neo4j_driver):
        """Test creating and using indexes in real database."""
        async with real_neo4j_driver.session() as session:
            # Create an index
            await session.run("""
                CREATE INDEX file_path_index IF NOT EXISTS
                FOR (f:CodeFile) ON (f.path)
            """)
            
            # Create fulltext index with whitespace analyzer
            await session.run("""
                CREATE FULLTEXT INDEX code_content_fulltext IF NOT EXISTS
                FOR (f:CodeFile) ON EACH [f.content]
                OPTIONS { indexConfig: { `fulltext.analyzer`: 'whitespace' } }
            """)
            
            # Wait for indexes to be online
            await asyncio.sleep(1)
            
            # Insert test data
            await session.run("""
                CREATE (f1:CodeFile {
                    path: '/api/auth.ts',
                    content: 'function authenticate(): Promise<void> { return login(); }'
                })
                CREATE (f2:CodeFile {
                    path: '/api/users.ts', 
                    content: 'async function getUser(id: number): Promise<User> { return db.find(id); }'
                })
            """)
            
            # Test fulltext search with special characters (no escaping needed with whitespace analyzer!)
            result = await session.run("""
                CALL db.index.fulltext.queryNodes('code_content_fulltext', 'Promise<void>')
                YIELD node, score
                RETURN node.path as path, score
                ORDER BY score DESC
            """)
            records = await result.data()
            
            assert len(records) > 0
            assert '/api/auth.ts' in [r['path'] for r in records]
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_transaction_rollback(self, real_neo4j_driver):
        """Test transaction rollback behavior in real database."""
        async with real_neo4j_driver.session() as session:
            # Start a transaction
            tx = await session.begin_transaction()
            
            try:
                # Create a node in transaction
                await tx.run("CREATE (n:TestNode {name: 'should_be_rolled_back'})")
                
                # Verify node exists in transaction
                result = await tx.run("MATCH (n:TestNode) RETURN count(n) as count")
                record = await result.single()
                assert record["count"] == 1
                
                # Rollback transaction
                await tx.rollback()
            except Exception:
                await tx.rollback()
                raise
            
            # Verify node doesn't exist after rollback
            result = await session.run("MATCH (n:TestNode) RETURN count(n) as count")
            record = await result.single()
            assert record["count"] == 0
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_queries(self, real_neo4j_driver):
        """Test executing concurrent queries against real database."""
        async with real_neo4j_driver.session() as session:
            # Create test data
            await session.run("""
                UNWIND range(1, 100) as i
                CREATE (n:Number {value: i})
            """)
        
        # Execute multiple concurrent queries
        async def run_query(driver, start, end):
            async with driver.session() as session:
                result = await session.run("""
                    MATCH (n:Number) 
                    WHERE n.value >= $start AND n.value <= $end
                    RETURN count(n) as count
                """, start=start, end=end)
                record = await result.single()
                return record["count"]
        
        # Run 10 concurrent queries
        tasks = [
            run_query(real_neo4j_driver, i*10+1, (i+1)*10)
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)
        
        # Each query should return 10 nodes
        assert all(count == 10 for count in results)
        assert sum(results) == 100
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_constraint_enforcement(self, real_neo4j_driver):
        """Test constraint enforcement in real database."""
        async with real_neo4j_driver.session() as session:
            # Create a uniqueness constraint
            await session.run("""
                CREATE CONSTRAINT unique_file_path IF NOT EXISTS
                FOR (f:CodeFile) REQUIRE f.path IS UNIQUE
            """)
            
            # Wait for constraint to be active
            await asyncio.sleep(1)
            
            # Insert first file
            await session.run("""
                CREATE (f:CodeFile {path: '/unique/file.py', content: 'first'})
            """)
            
            # Try to insert duplicate - should fail
            with pytest.raises(Neo4jError):
                await session.run("""
                    CREATE (f:CodeFile {path: '/unique/file.py', content: 'duplicate'})
                """)
            
            # Verify only one node exists
            result = await session.run("""
                MATCH (f:CodeFile {path: '/unique/file.py'})
                RETURN f.content as content
            """)
            record = await result.single()
            assert record["content"] == "first"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_graph_relationships(self, real_neo4j_driver):
        """Test creating and querying relationships in real database."""
        async with real_neo4j_driver.session() as session:
            # Create nodes with relationships
            await session.run("""
                CREATE (pkg:Package {name: 'main'})
                CREATE (mod1:Module {name: 'auth', path: '/auth.py'})
                CREATE (mod2:Module {name: 'users', path: '/users.py'})
                CREATE (func1:Function {name: 'login'})
                CREATE (func2:Function {name: 'get_user'})
                
                CREATE (pkg)-[:CONTAINS]->(mod1)
                CREATE (pkg)-[:CONTAINS]->(mod2)
                CREATE (mod1)-[:DEFINES]->(func1)
                CREATE (mod2)-[:DEFINES]->(func2)
                CREATE (mod2)-[:IMPORTS]->(mod1)
            """)
            
            # Query graph traversal
            result = await session.run("""
                MATCH path = (pkg:Package {name: 'main'})-[:CONTAINS*1..2]->()-[:DEFINES]->(f:Function)
                RETURN f.name as function_name
                ORDER BY function_name
            """)
            functions = [record["function_name"] async for record in result]
            
            assert functions == ['get_user', 'login']
            
            # Query relationship patterns
            result = await session.run("""
                MATCH (m1:Module)-[:IMPORTS]->(m2:Module)
                RETURN m1.name as importer, m2.name as imported
            """)
            record = await result.single()
            
            assert record["importer"] == "users"
            assert record["imported"] == "auth"


class TestRealNeo4jRAG:
    """Test Neo4jRAG with real database and embeddings."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_initialize_rag_with_real_database(self, real_neo4j_rag):
        """Test initializing Neo4jRAG with real database."""
        # RAG should be initialized by fixture
        assert real_neo4j_rag.neo4j_driver is not None
        assert real_neo4j_rag.project_name == "test_project"
        
        # Verify database is accessible
        async with real_neo4j_rag.neo4j_driver.session() as session:
            result = await session.run("RETURN 'initialized' as status")
            record = await result.single()
            assert record["status"] == "initialized"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_index_and_search_code(self, real_neo4j_rag):
        """Test indexing and searching code with real database."""
        # Create test code files
        test_files = [
            CodeFile(
                path="/src/auth.py",
                content="""
def authenticate(username: str, password: str) -> bool:
    '''Authenticate user with credentials'''
    if not username or not password:
        return False
    return check_credentials(username, password)
""",
                language="python"
            ),
            CodeFile(
                path="/src/database.py",
                content="""
async def get_user(user_id: int) -> Optional[User]:
    '''Fetch user from database by ID'''
    async with get_session() as session:
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()
""",
                language="python"
            ),
            CodeFile(
                path="/tests/test_auth.py",
                content="""
def test_authenticate_valid_credentials():
    assert authenticate('user', 'pass') == True

def test_authenticate_empty_credentials():
    assert authenticate('', '') == False
""",
                language="python"
            )
        ]
        
        # Index files
        for file in test_files:
            await real_neo4j_rag.index_file(file)
        
        # Search for authentication-related code
        results = await real_neo4j_rag.search_code("authentication user credentials")
        
        assert len(results) > 0
        # Auth-related files should be in results
        result_paths = [r.file_path for r in results]
        assert "/src/auth.py" in result_paths or "/tests/test_auth.py" in result_paths
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_vector_search_with_embeddings(self, real_neo4j_rag):
        """Test vector search functionality with real embeddings."""
        # Create and index test files
        test_files = [
            CodeFile(
                path="/api/payment.ts",
                content="async function processPayment(amount: number): Promise<PaymentResult> { /* payment logic */ }",
                language="typescript"
            ),
            CodeFile(
                path="/api/invoice.ts",
                content="function generateInvoice(items: Item[]): Invoice { /* invoice generation */ }",
                language="typescript"
            ),
            CodeFile(
                path="/utils/logger.ts",
                content="export function logError(error: Error): void { console.error(error); }",
                language="typescript"
            )
        ]
        
        for file in test_files:
            await real_neo4j_rag.index_file(file)
        
        # Search semantically for financial operations
        results = await real_neo4j_rag.search_code("financial transactions billing", use_vector=True)
        
        # Payment and invoice files should rank higher
        if results:
            top_paths = [r.file_path for r in results[:2]]
            assert any("payment" in path or "invoice" in path for path in top_paths)
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pattern_search_with_special_characters(self, real_neo4j_rag):
        """Test pattern search with special characters using real database."""
        # Create files with special characters in content
        test_files = [
            CodeFile(
                path="/src/types.ts",
                content="interface User<T> { id: number; data: T; }",
                language="typescript"
            ),
            CodeFile(
                path="/src/api.ts",
                content="function fetch(): Promise<void> { return Promise.resolve(); }",
                language="typescript"
            ),
            CodeFile(
                path="/src/array.ts",
                content="const items = array[index];",
                language="typescript"
            )
        ]
        
        for file in test_files:
            await real_neo4j_rag.index_file(file)
        
        # Search for patterns with special characters
        # With whitespace analyzer, these should work without escaping!
        test_patterns = [
            "Promise<void>",
            "User<T>",
            "array[index]"
        ]
        
        for pattern in test_patterns:
            results = await real_neo4j_rag.search_code(pattern, use_pattern=True)
            # Should find results without needing to escape
            assert len(results) > 0, f"Pattern '{pattern}' should return results"


class TestRealDatabasePerformance:
    """Test performance characteristics with real database."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.benchmark
    async def test_bulk_insert_performance(self, real_neo4j_driver):
        """Test bulk insert performance with real database."""
        import time
        
        async with real_neo4j_driver.session() as session:
            # Measure bulk insert time
            start_time = time.time()
            
            # Insert 1000 nodes in a single query
            await session.run("""
                UNWIND range(1, 1000) as i
                CREATE (n:PerfTest {
                    id: i,
                    name: 'Node_' + toString(i),
                    timestamp: timestamp(),
                    data: 'x' * 100
                })
            """)
            
            insert_time = time.time() - start_time
            
            # Verify all nodes were created
            result = await session.run("MATCH (n:PerfTest) RETURN count(n) as count")
            record = await result.single()
            assert record["count"] == 1000
            
            # Performance assertion - should complete in reasonable time
            assert insert_time < 5.0, f"Bulk insert took {insert_time:.2f}s, expected < 5s"
            
            print(f"Bulk insert of 1000 nodes: {insert_time:.3f}s ({1000/insert_time:.0f} nodes/sec)")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.benchmark
    async def test_query_performance_with_index(self, real_neo4j_driver):
        """Test query performance with and without indexes."""
        import time
        
        async with real_neo4j_driver.session() as session:
            # Create test data
            await session.run("""
                UNWIND range(1, 10000) as i
                CREATE (n:IndexTest {
                    id: i,
                    category: CASE WHEN i % 10 = 0 THEN 'special' ELSE 'normal' END,
                    value: rand() * 1000
                })
            """)
            
            # Query without index
            start_time = time.time()
            result = await session.run("""
                MATCH (n:IndexTest {category: 'special'})
                RETURN count(n) as count
            """)
            record = await result.single()
            no_index_time = time.time() - start_time
            assert record["count"] == 1000
            
            # Create index
            await session.run("""
                CREATE INDEX category_index IF NOT EXISTS
                FOR (n:IndexTest) ON (n.category)
            """)
            await asyncio.sleep(1)  # Wait for index to be online
            
            # Query with index
            start_time = time.time()
            result = await session.run("""
                MATCH (n:IndexTest {category: 'special'})
                RETURN count(n) as count
            """)
            record = await result.single()
            with_index_time = time.time() - start_time
            assert record["count"] == 1000
            
            print(f"Query without index: {no_index_time:.3f}s")
            print(f"Query with index: {with_index_time:.3f}s")
            print(f"Speedup: {no_index_time/with_index_time:.1f}x")
            
            # Index should provide significant speedup
            assert with_index_time < no_index_time


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "integration"])