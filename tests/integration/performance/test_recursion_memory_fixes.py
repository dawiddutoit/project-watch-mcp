"""Test recursion depth limits and memory management fixes.

This test suite validates:
1. Recursion depth protection in complexity analyzers
2. Memory-efficient batch processing in vector operations
3. Resource cleanup validation
"""

import asyncio
import pytest
import sys
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.project_watch_mcp.complexity_analysis.languages.java_analyzer import JavaComplexityAnalyzer
from src.project_watch_mcp.complexity_analysis.languages.kotlin_analyzer import KotlinComplexityAnalyzer
from src.project_watch_mcp.vector_search.neo4j_native_vectors import (
    NativeVectorIndex,
    VectorIndexConfig,
    VectorUpsertResult
)


class TestRecursionDepthProtection:
    """Test recursion depth limits prevent stack overflow."""
    
    @pytest.mark.asyncio
    async def test_java_analyzer_deep_nesting(self):
        """Test Java analyzer handles deeply nested code without stack overflow."""
        analyzer = JavaComplexityAnalyzer()
        
        # Create deeply nested code
        depth = 150  # More than MAX_DEPTH (100)
        nested_code = "class Test {\n"
        for i in range(depth):
            nested_code += f"  {'  ' * i}if (condition{i}) {{\n"
        for i in range(depth - 1, -1, -1):
            nested_code += f"  {'  ' * i}}}\n"
        nested_code += "}"
        
        # This should not cause stack overflow
        with patch('src.project_watch_mcp.complexity_analysis.languages.java_analyzer.logger') as mock_logger:
            result = await analyzer.analyze_code(nested_code)
            
            # Should have logged warnings about max depth
            assert mock_logger.warning.called
            warning_messages = [call[0][0] for call in mock_logger.warning.call_args_list]
            assert any("Maximum recursion depth" in msg for msg in warning_messages)
    
    @pytest.mark.asyncio
    async def test_kotlin_analyzer_deep_nesting(self):
        """Test Kotlin analyzer handles deeply nested code with protection."""
        analyzer = KotlinComplexityAnalyzer()
        
        # Create deeply nested code
        depth = 150  # More than max_allowed_depth (100)
        nested_code = "fun test() {\n"
        for i in range(depth):
            nested_code += f"{'  ' * i}if (condition{i}) {{\n"
        for i in range(depth - 1, -1, -1):
            nested_code += f"{'  ' * i}}}\n"
        nested_code += "}"
        
        # This should not cause issues and should cap depth
        with patch('src.project_watch_mcp.complexity_analysis.languages.kotlin_analyzer.logger') as mock_logger:
            result = await analyzer.analyze_code(nested_code)
            
            # Max depth should be capped at 100
            if result.functions:
                assert result.functions[0].depth <= 100
            
            # Should have logged warning if depth exceeded
            if mock_logger.warning.called:
                warning_messages = [call[0][0] for call in mock_logger.warning.call_args_list]
                assert any("Maximum nesting depth" in msg for msg in warning_messages)
    
    @pytest.mark.asyncio 
    async def test_java_analyzer_recursive_tree_walking(self):
        """Test tree walking methods have recursion protection."""
        analyzer = JavaComplexityAnalyzer()
        
        # Create a mock deeply nested AST node
        def create_nested_node(depth):
            node = MagicMock()
            node.type = 'block'
            node.start_point = (0, 0)
            node.end_point = (1, 0)
            
            if depth > 0:
                child = create_nested_node(depth - 1)
                node.children = [child]
            else:
                node.children = []
            
            return node
        
        # Create node deeper than MAX_DEPTH
        deep_node = create_nested_node(150)
        
        # Test _walk_tree_for_functions
        functions = []
        with patch('src.project_watch_mcp.complexity_analysis.languages.java_analyzer.logger') as mock_logger:
            analyzer._walk_tree_for_functions(deep_node, "test code", functions)
            
            # Should have hit recursion limit
            assert mock_logger.warning.called
            warning_messages = [call[0][0] for call in mock_logger.warning.call_args_list]
            assert any("Maximum recursion depth" in msg and "function extraction" in msg 
                      for msg in warning_messages)
        
        # Test _walk_tree_for_classes
        classes = []
        with patch('src.project_watch_mcp.complexity_analysis.languages.java_analyzer.logger') as mock_logger:
            analyzer._walk_tree_for_classes(deep_node, "test code", classes)
            
            # Should have hit recursion limit
            assert mock_logger.warning.called
            warning_messages = [call[0][0] for call in mock_logger.warning.call_args_list]
            assert any("Maximum recursion depth" in msg and "class extraction" in msg 
                      for msg in warning_messages)


class TestMemoryManagement:
    """Test memory-efficient batch processing."""
    
    @pytest.mark.asyncio
    async def test_batch_vector_upsert_chunks(self):
        """Test batch upsert processes vectors in memory-efficient chunks."""
        # Create mock driver
        mock_driver = AsyncMock()
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        
        # Setup mock chain - mock_driver.session() returns an async context manager
        mock_driver.session.return_value = mock_session
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session.run.return_value = mock_result
        
        # Mock async iteration for results
        processed_records = [{"node_id": f"node_{i}"} for i in range(2000)]
        
        async def mock_aiter():
            for record in processed_records:
                yield record
        
        mock_result.__aiter__ = mock_aiter
        
        # Create vector index
        config = VectorIndexConfig(dimensions=128)
        index = NativeVectorIndex(mock_driver, config)
        
        # Create large batch of vectors
        large_batch = [
            {
                "node_id": f"node_{i}",
                "vector": [0.1] * 128,
                "metadata": {"index": i}
            }
            for i in range(2000)
        ]
        
        # Process batch
        results = await index.batch_upsert_vectors(large_batch, batch_size=500)
        
        # Should have processed all vectors
        assert len(results) == 2000
        
        # Check that session.run was called multiple times (chunking)
        # With batch_size=500 and 2000 vectors, should be 4 calls
        assert mock_session.run.call_count == 4
        
        # Verify each chunk size
        for call in mock_session.run.call_args_list:
            batch_param = call[0][1]["batch"]
            assert len(batch_param) <= 500
    
    @pytest.mark.asyncio
    async def test_batch_vector_memory_cleanup(self):
        """Test that batch processing cleans up memory after each chunk."""
        mock_driver = AsyncMock()
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        
        mock_driver.session.return_value = mock_session
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session.run.return_value = mock_result
        
        # Mock empty results
        async def mock_aiter():
            return
            yield  # Make it an async generator
        
        mock_result.__aiter__ = mock_aiter
        
        config = VectorIndexConfig(dimensions=128)
        index = NativeVectorIndex(mock_driver, config)
        
        # Create batch with some invalid vectors
        mixed_batch = [
            {
                "node_id": f"valid_{i}",
                "vector": [0.1] * 128,
                "metadata": {"valid": True}
            } if i % 2 == 0 else {
                "node_id": f"invalid_{i}",
                "vector": [0.1] * 64,  # Wrong dimensions
                "metadata": {"valid": False}
            }
            for i in range(100)
        ]
        
        # Process batch
        results = await index.batch_upsert_vectors(mixed_batch, batch_size=25)
        
        # Should have results for all items
        assert len(results) == 100
        
        # Check invalid vectors were marked as failed
        failed_results = [r for r in results if not r.success]
        assert len(failed_results) == 50  # Half should fail due to wrong dimensions
        
        # Verify error messages
        for result in failed_results:
            assert "dimension mismatch" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_large_batch_doesnt_accumulate_memory(self):
        """Test that very large batches don't accumulate unbounded memory."""
        mock_driver = AsyncMock()
        mock_session = AsyncMock()
        
        mock_driver.session.return_value = mock_session
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        
        # Track memory usage through batch data sizes
        batch_sizes_seen = []
        
        async def track_batch_size(query, params):
            batch_sizes_seen.append(len(params.get("batch", [])))
            mock_result = AsyncMock()
            
            async def mock_aiter():
                for item in params.get("batch", []):
                    yield {"node_id": item["node_id"]}
            
            mock_result.__aiter__ = mock_aiter
            return mock_result
        
        mock_session.run = track_batch_size
        
        config = VectorIndexConfig(dimensions=128)
        index = NativeVectorIndex(mock_driver, config)
        
        # Create very large batch
        huge_batch = [
            {
                "node_id": f"node_{i}",
                "vector": [0.1] * 128,
                "metadata": {"index": i, "data": "x" * 1000}  # Add some bulk
            }
            for i in range(10000)
        ]
        
        # Process with smaller chunk size
        results = await index.batch_upsert_vectors(huge_batch, batch_size=100)
        
        # Verify chunking happened
        assert len(batch_sizes_seen) == 100  # 10000 / 100
        
        # Verify no single batch exceeded the limit
        assert all(size <= 100 for size in batch_sizes_seen)
        
        # All items should be processed
        assert len(results) == 10000


class TestResourceCleanup:
    """Test proper resource cleanup in operations."""
    
    @pytest.mark.asyncio
    async def test_vector_index_session_cleanup(self):
        """Test that database sessions are properly closed."""
        mock_driver = AsyncMock()
        mock_session = AsyncMock()
        
        mock_driver.session.return_value = mock_session
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        
        config = VectorIndexConfig(dimensions=128)
        index = NativeVectorIndex(mock_driver, config)
        
        # Test various operations ensure session cleanup
        
        # Create index
        await index.create_index()
        assert mock_session.__aexit__.called
        
        # Reset mock
        mock_session.__aexit__.reset_mock()
        
        # Search operation
        try:
            await index.search([0.1] * 128, top_k=5)
        except:
            pass  # Expected to fail with mock
        assert mock_session.__aexit__.called
        
        # Reset mock
        mock_session.__aexit__.reset_mock()
        
        # Stats operation
        try:
            await index.get_index_stats()
        except:
            pass  # Expected to fail with mock
        assert mock_session.__aexit__.called
    
    @pytest.mark.asyncio
    async def test_analyzer_error_handling_cleanup(self):
        """Test analyzers clean up properly on errors."""
        # Test Java analyzer - skip if tree-sitter not available
        java_analyzer = JavaComplexityAnalyzer()
        
        if java_analyzer._parser is not None:
            # Invalid Java code
            invalid_java = "class { this is not valid java }"
            
            result = await java_analyzer.analyze_code(invalid_java)
            
            # Should return result without crash (may or may not have error)
            assert result is not None
            assert result.summary is not None
        
        # Test Kotlin analyzer
        kotlin_analyzer = KotlinComplexityAnalyzer()
        
        # Invalid Kotlin code with mismatched braces
        invalid_kotlin = "fun test() { {{ } }"
        
        result = await kotlin_analyzer.analyze_code(invalid_kotlin)
        
        # Should return error result for invalid syntax
        assert result.error is not None
        assert "Syntax error" in result.error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])