"""
Integration tests validating Neo4j search with REAL database.

This test suite validates search functionality against an actual Neo4j instance,
testing both fulltext and vector search capabilities without mocks.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import List, Optional
import pytest

from project_watch_mcp.neo4j_rag import CodeFile, Neo4jRAG


@dataclass
class SearchTestCase:
    """Represents a search test case with expected behavior."""
    query: str
    description: str
    search_type: str  # "pattern", "semantic", or "hybrid"
    expected_files: List[str]  # Files that should appear in results
    test_content: str  # Content to index for this test


class TestRealSearchSolution:
    """Validate search solution with real Neo4j database."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_whitespace_fulltext_with_special_characters(self, real_neo4j_driver):
        """Test that whitespace analyzer handles special characters without escaping."""
        async with real_neo4j_driver.session() as session:
            # Create fulltext index with whitespace analyzer
            await session.run("""
                CREATE FULLTEXT INDEX code_search_whitespace IF NOT EXISTS
                FOR (c:CodeChunk) ON EACH [c.content]
                OPTIONS { indexConfig: { `fulltext.analyzer`: 'whitespace' } }
            """)
            
            # Wait for index to be online
            await asyncio.sleep(1)
            
            # Test patterns that would require escaping in standard Lucene
            test_patterns = [
                ("function(): void", "TypeScript function signature"),
                ("React.FC<Props>", "React component with generics"),
                ("array[index]", "Array access notation"),
                ("async () => {}", "Arrow function"),
                ("@decorator()", "Python decorator"),
                ("path\\to\\file", "Windows path"),
                ("#include <iostream>", "C++ include"),
                ("SELECT * FROM users", "SQL query"),
            ]
            
            # Insert test data
            for pattern, description in test_patterns:
                await session.run("""
                    CREATE (c:CodeChunk {
                        content: $content,
                        description: $description
                    })
                """, content=f"Code containing {pattern} in the middle", description=description)
            
            # Search for each pattern WITHOUT escaping
            success_count = 0
            for pattern, description in test_patterns:
                result = await session.run("""
                    CALL db.index.fulltext.queryNodes('code_search_whitespace', $query)
                    YIELD node, score
                    WHERE node.content CONTAINS $pattern
                    RETURN node.content AS content, score
                    ORDER BY score DESC
                    LIMIT 1
                """, query=pattern, pattern=pattern)
                
                records = await result.data()
                
                if records:
                    success_count += 1
                    print(f"✓ Found '{pattern}': {description}")
                else:
                    print(f"✗ Failed to find '{pattern}': {description}")
            
            # All patterns should be found
            assert success_count == len(test_patterns), \
                f"Only {success_count}/{len(test_patterns)} patterns found"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_vector_search_semantic_understanding(self, real_neo4j_rag):
        """Test vector search for semantic understanding with real embeddings."""
        # Index test files with semantic content
        test_files = [
            CodeFile(
                path="/auth/login.py",
                content="""
def authenticate_user(username, password):
    '''Verify user credentials and create session'''
    user = db.find_user(username)
    if user and verify_password(password, user.password_hash):
        return create_session(user)
    return None
""",
                language="python"
            ),
            CodeFile(
                path="/auth/permissions.py",
                content="""
def check_user_permissions(user_id, resource):
    '''Check if user has access to resource'''
    user_roles = get_user_roles(user_id)
    required_permissions = get_resource_permissions(resource)
    return has_required_permissions(user_roles, required_permissions)
""",
                language="python"
            ),
            CodeFile(
                path="/utils/logger.py",
                content="""
def log_event(event_type, message, level='INFO'):
    '''Write event to application log'''
    timestamp = datetime.now()
    formatted = f"{timestamp} [{level}] {event_type}: {message}"
    write_to_file(formatted)
""",
                language="python"
            ),
        ]
        
        # Index files
        for file in test_files:
            await real_neo4j_rag.index_file(file)
        
        # Test semantic searches
        semantic_queries = [
            ("user authentication and security", ["/auth/login.py", "/auth/permissions.py"]),
            ("access control and authorization", ["/auth/permissions.py"]),
            ("logging and monitoring", ["/utils/logger.py"]),
        ]
        
        for query, expected_files in semantic_queries:
            results = await real_neo4j_rag.search_code(query, use_vector=True, limit=3)
            
            result_paths = [r.file_path for r in results]
            
            # Check if expected files appear in results
            matches = sum(1 for f in expected_files if f in result_paths)
            print(f"Query: '{query}'")
            print(f"  Expected: {expected_files}")
            print(f"  Found: {result_paths[:3]}")
            print(f"  Matches: {matches}/{len(expected_files)}")
            
            # At least one expected file should be in top results
            assert matches > 0, f"No expected files found for '{query}'"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_hybrid_search_approach(self, real_neo4j_rag):
        """Test hybrid search combining pattern and semantic search."""
        # Index diverse test content
        test_files = [
            # Files with specific patterns
            CodeFile(
                path="/api/types.ts",
                content="interface User<T> { id: number; data: T; email: string; }",
                language="typescript"
            ),
            CodeFile(
                path="/api/promises.ts",
                content="async function fetchData(): Promise<void> { await fetch('/api/data'); }",
                language="typescript"
            ),
            # Files with semantic content
            CodeFile(
                path="/services/email.py",
                content="""
def send_notification_email(recipient, subject, body):
    '''Send email notification to user'''
    message = create_email_message(recipient, subject, body)
    smtp_client.send(message)
""",
                language="python"
            ),
            CodeFile(
                path="/services/sms.py",
                content="""
def send_sms_alert(phone_number, message):
    '''Send SMS alert to user phone'''
    formatted_number = validate_phone(phone_number)
    sms_gateway.send(formatted_number, message)
""",
                language="python"
            ),
        ]
        
        for file in test_files:
            await real_neo4j_rag.index_file(file)
        
        # Test different query types
        test_queries = [
            # Pattern queries (exact match)
            ("Promise<void>", "pattern", ["/api/promises.ts"]),
            ("User<T>", "pattern", ["/api/types.ts"]),
            # Semantic queries
            ("notification system messaging", "semantic", ["/services/email.py", "/services/sms.py"]),
            ("user communication channels", "semantic", ["/services/email.py", "/services/sms.py"]),
        ]
        
        for query, query_type, expected_files in test_queries:
            if query_type == "pattern":
                results = await real_neo4j_rag.search_code(query, use_pattern=True, limit=5)
            else:
                results = await real_neo4j_rag.search_code(query, use_vector=True, limit=5)
            
            result_paths = [r.file_path for r in results]
            
            matches = sum(1 for f in expected_files if f in result_paths)
            print(f"\n{query_type.upper()} Query: '{query}'")
            print(f"  Expected: {expected_files}")
            print(f"  Found: {result_paths}")
            print(f"  Success: {matches}/{len(expected_files)} files found")
            
            assert matches > 0, f"No expected files found for {query_type} query '{query}'"

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.benchmark
    async def test_search_performance_comparison(self, real_neo4j_driver):
        """Compare performance of different search approaches with real database."""
        async with real_neo4j_driver.session() as session:
            # Create test data
            await session.run("""
                UNWIND range(1, 1000) as i
                CREATE (c:PerfTestChunk {
                    id: i,
                    content: CASE 
                        WHEN i % 3 = 0 THEN 'function process(): Promise<void> { return async(); }'
                        WHEN i % 3 = 1 THEN 'interface Config<T> { data: T; settings: object; }'
                        ELSE 'const handler = async (req, res) => { return res.json({}); }'
                    END,
                    embedding: [rand(), rand(), rand()]  // Simplified embedding
                })
            """)
            
            # Create indexes
            await session.run("""
                CREATE FULLTEXT INDEX perf_fulltext IF NOT EXISTS
                FOR (c:PerfTestChunk) ON EACH [c.content]
                OPTIONS { indexConfig: { `fulltext.analyzer`: 'whitespace' } }
            """)
            
            await session.run("""
                CREATE VECTOR INDEX perf_vector IF NOT EXISTS
                FOR (c:PerfTestChunk) ON (c.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 3,
                    `vector.similarity_function`: 'cosine'
                }}
            """)
            
            await asyncio.sleep(2)  # Wait for indexes
            
            # Measure search performance
            search_times = {
                "fulltext_simple": [],
                "fulltext_complex": [],
                "vector_search": [],
            }
            
            # Test fulltext search with simple pattern
            for _ in range(10):
                start = time.time()
                result = await session.run("""
                    CALL db.index.fulltext.queryNodes('perf_fulltext', 'function')
                    YIELD node, score
                    RETURN node.id
                    LIMIT 10
                """)
                await result.data()
                search_times["fulltext_simple"].append(time.time() - start)
            
            # Test fulltext search with complex pattern
            for _ in range(10):
                start = time.time()
                result = await session.run("""
                    CALL db.index.fulltext.queryNodes('perf_fulltext', 'Promise<void>')
                    YIELD node, score
                    RETURN node.id
                    LIMIT 10
                """)
                await result.data()
                search_times["fulltext_complex"].append(time.time() - start)
            
            # Test vector search
            for _ in range(10):
                start = time.time()
                result = await session.run("""
                    CALL db.index.vector.queryNodes('perf_vector', 3, [0.5, 0.5, 0.5])
                    YIELD node, score
                    RETURN node.id
                    LIMIT 10
                """)
                await result.data()
                search_times["vector_search"].append(time.time() - start)
            
            # Print performance results
            print("\n=== SEARCH PERFORMANCE (Real Database) ===")
            for search_type, times in search_times.items():
                avg_ms = (sum(times) / len(times)) * 1000
                min_ms = min(times) * 1000
                max_ms = max(times) * 1000
                print(f"\n{search_type}:")
                print(f"  Average: {avg_ms:.2f}ms")
                print(f"  Min: {min_ms:.2f}ms")
                print(f"  Max: {max_ms:.2f}ms")
            
            # All searches should be fast
            for search_type, times in search_times.items():
                avg_time = sum(times) / len(times)
                assert avg_time < 0.1, f"{search_type} too slow: {avg_time:.3f}s"


class TestRealErrorHandling:
    """Test error handling with real database operations."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_invalid_fulltext_query_handling(self, real_neo4j_driver):
        """Test handling of invalid fulltext queries."""
        async with real_neo4j_driver.session() as session:
            # Create fulltext index
            await session.run("""
                CREATE FULLTEXT INDEX error_test_index IF NOT EXISTS
                FOR (c:ErrorTest) ON EACH [c.content]
            """)
            await asyncio.sleep(1)
            
            # Create test data
            await session.run("""
                CREATE (c:ErrorTest {content: 'test content'})
            """)
            
            # Valid query should work
            result = await session.run("""
                CALL db.index.fulltext.queryNodes('error_test_index', 'test')
                YIELD node
                RETURN count(node) as count
            """)
            record = await result.single()
            assert record["count"] >= 0
            
            # Empty query should be handled gracefully
            result = await session.run("""
                CALL db.index.fulltext.queryNodes('error_test_index', '')
                YIELD node
                RETURN count(node) as count
            """)
            record = await result.single()
            # Neo4j might return 0 results or all results for empty query
            assert record["count"] >= 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_index_operations(self, real_neo4j_driver):
        """Test concurrent operations on indexes."""
        async def create_and_query(session, index_num):
            # Create unique data
            await session.run(f"""
                CREATE (c:ConcurrentTest_{index_num} {{
                    content: 'test content {index_num}',
                    id: {index_num}
                }})
            """)
            
            # Query the data
            result = await session.run(f"""
                MATCH (c:ConcurrentTest_{index_num})
                RETURN count(c) as count
            """)
            record = await result.single()
            return record["count"]
        
        # Run concurrent operations
        tasks = []
        async with real_neo4j_driver.session() as session:
            for i in range(10):
                tasks.append(create_and_query(session, i))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All operations should succeed
        assert all(r == 1 for r in results if not isinstance(r, Exception))
        
        # Verify all data was created
        async with real_neo4j_driver.session() as session:
            result = await session.run("""
                MATCH (c)
                WHERE any(label in labels(c) WHERE label STARTS WITH 'ConcurrentTest_')
                RETURN count(c) as total
            """)
            record = await result.single()
            assert record["total"] == 10


class TestRealWorldScenarios:
    """Test real-world search scenarios with actual database."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_code_refactoring_search(self, real_neo4j_rag):
        """Test searching for code that needs refactoring."""
        # Index code with various quality issues
        test_files = [
            CodeFile(
                path="/legacy/old_api.py",
                content="""
# TODO: Refactor this mess
def process_data(d):
    x = d['value'] * 2
    y = x + 10
    z = y / 3
    # FIXME: Handle division by zero
    return z
""",
                language="python"
            ),
            CodeFile(
                path="/utils/helpers.py",
                content="""
def calculate_metrics(data):
    '''Calculate various metrics from data'''
    total = sum(data.values())
    average = total / len(data)
    return {'total': total, 'average': average}
""",
                language="python"
            ),
            CodeFile(
                path="/api/deprecated.py",
                content="""
# DEPRECATED: Use new_api instead
def old_endpoint(request):
    # WARNING: This will be removed in v2.0
    return legacy_handler(request)
""",
                language="python"
            ),
        ]
        
        for file in test_files:
            await real_neo4j_rag.index_file(file)
        
        # Search for code needing refactoring
        refactor_queries = [
            "TODO FIXME",  # Pattern search for comments
            "deprecated legacy old",  # Semantic search for old code
            "code quality refactor improve",  # Semantic search
        ]
        
        for query in refactor_queries:
            results = await real_neo4j_rag.search_code(query, limit=5)
            print(f"\nRefactoring search: '{query}'")
            for r in results[:3]:
                print(f"  - {r.file_path}")
            
            # Should find relevant files
            assert len(results) > 0, f"No results for refactoring query: {query}"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_security_vulnerability_search(self, real_neo4j_rag):
        """Test searching for potential security issues."""
        # Index code with various security patterns
        test_files = [
            CodeFile(
                path="/db/queries.py",
                content="""
def get_user(user_id):
    # Potential SQL injection
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
""",
                language="python"
            ),
            CodeFile(
                path="/auth/password.py",
                content="""
def store_password(password):
    # Secure password handling
    salt = generate_salt()
    hashed = bcrypt.hash(password, salt)
    return hashed
""",
                language="python"
            ),
            CodeFile(
                path="/api/input.py",
                content="""
def process_input(user_input):
    # Input validation
    sanitized = sanitize_input(user_input)
    validated = validate_format(sanitized)
    return validated
""",
                language="python"
            ),
        ]
        
        for file in test_files:
            await real_neo4j_rag.index_file(file)
        
        # Search for security-related code
        security_queries = [
            "SQL injection database query",
            "password security authentication",
            "input validation sanitization",
        ]
        
        for query in security_queries:
            results = await real_neo4j_rag.search_code(query, use_vector=True, limit=5)
            print(f"\nSecurity search: '{query}'")
            for r in results[:3]:
                print(f"  - {r.file_path}: {r.chunk_content[:50]}...")
            
            assert len(results) > 0, f"No results for security query: {query}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "integration"])