"""Simple tests for resource management fixes without complex mocking."""

import asyncio
import pytest
import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.project_watch_mcp.complexity_analysis.languages.java_analyzer import JavaComplexityAnalyzer
from src.project_watch_mcp.complexity_analysis.languages.kotlin_analyzer import KotlinComplexityAnalyzer
from src.project_watch_mcp.vector_search.neo4j_native_vectors import (
    VectorIndexConfig,
    VectorUpsertResult
)


class TestComplexityAnalyzersRobustness:
    """Test that complexity analyzers handle edge cases robustly."""
    
    @pytest.mark.asyncio
    async def test_java_deeply_nested_classes(self):
        """Test Java analyzer handles deeply nested inner classes."""
        analyzer = JavaComplexityAnalyzer()
        
        # Create code with nested classes
        code = """
        public class Outer {
            class Inner1 {
                class Inner2 {
                    class Inner3 {
                        class Inner4 {
                            class Inner5 {
                                void deepMethod() {
                                    if (true) {
                                        while (true) {
                                            for (int i = 0; i < 10; i++) {
                                                // Deep nesting
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        """
        
        result = await analyzer.analyze_code(code)
        
        # Should handle without crash
        assert result is not None
        assert result.summary is not None
        
        # If parser is available, should have found classes
        if analyzer._parser and not result.error:
            assert result.summary.class_count > 0
    
    @pytest.mark.asyncio
    async def test_kotlin_extreme_nesting(self):
        """Test Kotlin analyzer handles extreme nesting levels."""
        analyzer = KotlinComplexityAnalyzer()
        
        # Create code with extreme nesting
        code = """
        fun extremeNesting() {
            if (a) {
                if (b) {
                    if (c) {
                        when(x) {
                            1 -> {
                                for (i in 1..10) {
                                    while (true) {
                                        do {
                                            if (nested) {
                                                // Very deep
                                                val lambda = { x: Int ->
                                                    x * 2
                                                }
                                            }
                                        } while (false)
                                    }
                                }
                            }
                            2 -> println("two")
                        }
                    }
                }
            }
        }
        """
        
        result = await analyzer.analyze_code(code)
        
        # Should handle without crash
        assert result is not None
        assert result.summary is not None
        assert result.summary.total_complexity > 0
        
        # Should have found the function
        assert len(result.functions) > 0
        assert result.functions[0].depth <= 100  # Should be capped
    
    @pytest.mark.asyncio
    async def test_java_recursive_method_calls(self):
        """Test Java analyzer handles recursive method patterns."""
        analyzer = JavaComplexityAnalyzer()
        
        code = """
        public class RecursiveExample {
            public int factorial(int n) {
                if (n <= 1) {
                    return 1;
                }
                return n * factorial(n - 1);
            }
            
            public void indirectRecursionA() {
                indirectRecursionB();
            }
            
            public void indirectRecursionB() {
                indirectRecursionA();
            }
        }
        """
        
        result = await analyzer.analyze_code(code)
        
        # Should analyze without infinite loops
        assert result is not None
        assert result.summary is not None
    
    @pytest.mark.asyncio
    async def test_kotlin_sealed_class_complexity(self):
        """Test Kotlin analyzer correctly handles sealed classes."""
        analyzer = KotlinComplexityAnalyzer()
        
        code = """
        sealed class Result {
            data class Success(val data: String) : Result()
            data class Error(val message: String) : Result()
            object Loading : Result()
        }
        
        fun handleResult(result: Result) {
            when (result) {
                is Result.Success -> println(result.data)
                is Result.Error -> println(result.message)
                Result.Loading -> println("Loading...")
            }
        }
        """
        
        result = await analyzer.analyze_code(code)
        
        # Should handle sealed classes
        assert result is not None
        assert result.summary is not None
        
        # Should find the sealed class
        if result.classes:
            sealed_class = next((c for c in result.classes if c.name == "Result"), None)
            if sealed_class:
                # Sealed class with subclasses should have added complexity
                assert sealed_class.total_complexity >= 1


class TestVectorBatchProcessing:
    """Test vector batch processing efficiency."""
    
    def test_vector_config_validation(self):
        """Test vector index configuration validation."""
        # Valid config
        config = VectorIndexConfig(dimensions=1536, provider="openai")
        assert config.dimensions == 1536
        assert config.provider == "openai"
        
        # Invalid similarity metric
        with pytest.raises(ValueError, match="Invalid similarity metric"):
            VectorIndexConfig(similarity_metric="invalid")
        
        # Invalid dimensions
        with pytest.raises(ValueError, match="Dimensions must be positive"):
            VectorIndexConfig(dimensions=0)
        
        # Warning for unusual dimensions (no exception, just warning)
        config = VectorIndexConfig(dimensions=999, provider="openai")
        assert config.dimensions == 999  # Should still work
    
    def test_vector_search_result_sorting(self):
        """Test that vector search results can be sorted by score."""
        from src.project_watch_mcp.vector_search.neo4j_native_vectors import VectorSearchResult
        
        results = [
            VectorSearchResult(node_id="1", score=0.5, metadata={}),
            VectorSearchResult(node_id="2", score=0.9, metadata={}),
            VectorSearchResult(node_id="3", score=0.7, metadata={}),
        ]
        
        sorted_results = sorted(results, reverse=True)
        
        # Should be sorted by score descending
        assert sorted_results[0].score == 0.9
        assert sorted_results[1].score == 0.7
        assert sorted_results[2].score == 0.5
    
    def test_vector_upsert_result_creation(self):
        """Test vector upsert result creation and attributes."""
        result = VectorUpsertResult(
            node_id="test_node",
            success=True,
            operation="created"
        )
        
        assert result.node_id == "test_node"
        assert result.success is True
        assert result.operation == "created"
        assert result.error is None
        
        # With error
        error_result = VectorUpsertResult(
            node_id="error_node",
            success=False,
            operation="failed",
            error="Test error message"
        )
        
        assert error_result.success is False
        assert error_result.error == "Test error message"


class TestMemoryEfficiency:
    """Test memory-efficient operations."""
    
    def test_batch_size_parameter(self):
        """Test that batch_size parameter exists and has reasonable default."""
        from inspect import signature
        from src.project_watch_mcp.vector_search.neo4j_native_vectors import NativeVectorIndex
        
        # Check method signature
        sig = signature(NativeVectorIndex.batch_upsert_vectors)
        params = sig.parameters
        
        # Should have batch_size parameter
        assert 'batch_size' in params
        
        # Should have default value
        assert params['batch_size'].default == 1000
    
    @pytest.mark.asyncio
    async def test_java_max_recursion_constant(self):
        """Test that Java analyzer has MAX_RECURSION constant."""
        analyzer = JavaComplexityAnalyzer()
        
        # Create a simple mock node to test _calculate_max_depth
        class MockNode:
            def __init__(self, depth_level):
                self.type = 'block' if depth_level % 2 == 0 else 'statement'
                self.children = []
                if depth_level > 0:
                    self.children = [MockNode(depth_level - 1)]
        
        # Create a very deep node structure
        deep_node = MockNode(200)
        
        # This should not crash due to recursion limit
        max_depth = analyzer._calculate_max_depth(deep_node, 0, 0)
        
        # Should be limited by MAX_RECURSION
        assert max_depth <= 100  # Based on our implementation
    
    @pytest.mark.asyncio
    async def test_kotlin_max_nesting_protection(self):
        """Test that Kotlin analyzer protects against excessive nesting."""
        analyzer = KotlinComplexityAnalyzer()
        
        # Create code with 200 nested blocks
        code = "{" * 200 + "}" * 200
        
        # Should not crash and should cap depth
        depth = analyzer._calculate_max_nesting(code)
        
        # Should be capped at max_allowed_depth (100)
        assert depth == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])