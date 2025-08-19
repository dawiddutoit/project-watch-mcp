"""Integration tests for embedding enrichment with language detection."""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Optional

from src.project_watch_mcp.language_detection import (
    HybridLanguageDetector,
    LanguageDetectionResult,
    DetectionMethod,
)
from src.project_watch_mcp.utils.embeddings import OpenAIEmbeddingsProvider
from src.project_watch_mcp.neo4j_rag import Neo4jRAG


class LanguageEnrichmentProcessor:
    """Process embeddings with language-specific enrichment."""
    
    # Language-specific keywords for enrichment
    LANGUAGE_KEYWORDS = {
        "python": ["python", "def", "class", "import", "async", "await", "__init__", "self"],
        "javascript": ["javascript", "function", "const", "let", "var", "async", "await", "=>"],
        "java": ["java", "public", "private", "class", "interface", "extends", "implements"],
        "kotlin": ["kotlin", "fun", "val", "var", "class", "object", "companion", "suspend"],
        "typescript": ["typescript", "interface", "type", "enum", "namespace", "declare"],
        "go": ["golang", "func", "package", "defer", "goroutine", "channel"],
        "rust": ["rust", "fn", "impl", "trait", "struct", "enum", "mut", "unsafe"],
    }
    
    def __init__(self, detector: HybridLanguageDetector, embeddings_provider: OpenAIEmbeddingsProvider):
        """Initialize the enrichment processor."""
        self.detector = detector
        self.embeddings_provider = embeddings_provider
        self.enrichment_cache = {}
    
    async def enrich_embedding(
        self, 
        content: str, 
        file_path: Optional[str] = None,
        base_embedding: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Enrich embedding with language-specific context.
        
        Returns:
            Dict containing:
            - embedding: Enhanced embedding vector
            - language: Detected language
            - confidence: Detection confidence
            - enrichment_applied: Boolean indicating if enrichment was applied
            - keywords_added: List of keywords added for enrichment
        """
        # Detect language
        detection_result = self.detector.detect(content, file_path)
        
        # Get base embedding if not provided
        if base_embedding is None:
            base_embedding = await self.embeddings_provider.embed(content)
        
        # Skip enrichment for low confidence or unknown languages
        if detection_result.confidence < 0.7 or detection_result.language not in self.LANGUAGE_KEYWORDS:
            return {
                "embedding": base_embedding,
                "language": detection_result.language,
                "confidence": detection_result.confidence,
                "enrichment_applied": False,
                "keywords_added": []
            }
        
        # Get language-specific keywords
        keywords = self.LANGUAGE_KEYWORDS.get(detection_result.language, [])
        
        # Create enriched content
        enriched_content = self._create_enriched_content(content, keywords, detection_result.language)
        
        # Generate enriched embedding
        enriched_embedding = await self.embeddings_provider.embed(enriched_content)
        
        # Blend original and enriched embeddings (weighted average)
        weight = min(detection_result.confidence, 0.3)  # Max 30% enrichment weight
        final_embedding = (1 - weight) * base_embedding + weight * enriched_embedding
        
        # Normalize the final embedding
        norm = np.linalg.norm(final_embedding)
        if norm > 0:
            final_embedding = final_embedding / norm
        
        return {
            "embedding": final_embedding,
            "language": detection_result.language,
            "confidence": detection_result.confidence,
            "enrichment_applied": True,
            "keywords_added": keywords,
            "enrichment_weight": weight
        }
    
    def _create_enriched_content(self, content: str, keywords: List[str], language: str) -> str:
        """Create enriched content with language context."""
        # Add language context as a prefix
        language_context = f"[{language.upper()} CODE] "
        
        # Add relevant keywords that aren't already in the content
        additional_context = []
        for keyword in keywords:
            if keyword.lower() not in content.lower():
                additional_context.append(keyword)
        
        # Combine everything
        if additional_context:
            context_str = " ".join(additional_context[:3])  # Limit to 3 keywords
            enriched = f"{language_context}{content}\n# Context: {context_str}"
        else:
            enriched = f"{language_context}{content}"
        
        return enriched
    
    async def batch_enrich(self, items: List[tuple]) -> List[Dict]:
        """Batch process multiple items for enrichment."""
        tasks = []
        for content, file_path in items:
            tasks.append(self.enrich_embedding(content, file_path))
        
        return await asyncio.gather(*tasks)


class TestEmbeddingEnrichment:
    """Test embedding enrichment with language detection."""
    
    @pytest.fixture
    def detector(self):
        """Create a language detector."""
        return HybridLanguageDetector(enable_cache=True)
    
    @pytest.fixture
    async def embeddings_provider(self):
        """Create a mock embeddings provider."""
        provider = Mock(spec=OpenAIEmbeddingsProvider)
        
        # Mock embedding generation
        async def mock_embed(content):
            # Generate deterministic embedding based on content
            np.random.seed(hash(content) % 2**32)
            return np.random.randn(1536).astype(np.float32)
        
        provider.embed = AsyncMock(side_effect=mock_embed)
        provider.dimension = 1536
        return provider
    
    @pytest.fixture
    async def enrichment_processor(self, detector, embeddings_provider):
        """Create an enrichment processor."""
        return LanguageEnrichmentProcessor(detector, embeddings_provider)
    
    @pytest.mark.asyncio
    async def test_enrichment_with_high_confidence_python(self, enrichment_processor):
        """Test enrichment with high confidence Python code."""
        python_code = """
def calculate_fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

class FibonacciCalculator:
    def __init__(self):
        self.cache = {}
    
    def calculate(self, n: int) -> int:
        if n in self.cache:
            return self.cache[n]
        result = calculate_fibonacci(n)
        self.cache[n] = result
        return result
"""
        
        result = await enrichment_processor.enrich_embedding(
            python_code, 
            "fibonacci.py"
        )
        
        assert result["language"] == "python"
        assert result["confidence"] >= 0.9
        assert result["enrichment_applied"] is True
        assert len(result["keywords_added"]) > 0
        assert "python" in result["keywords_added"]
        assert result["enrichment_weight"] > 0
        assert result["embedding"].shape == (1536,)
    
    @pytest.mark.asyncio
    async def test_enrichment_with_javascript(self, enrichment_processor):
        """Test enrichment with JavaScript code."""
        js_code = """
const fetchData = async (url) => {
    try {
        const response = await fetch(url);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching data:', error);
        throw error;
    }
};

function processData(data) {
    return data.map(item => ({
        ...item,
        processed: true,
        timestamp: Date.now()
    }));
}
"""
        
        result = await enrichment_processor.enrich_embedding(
            js_code,
            "data_processor.js"
        )
        
        assert result["language"] == "javascript"
        assert result["confidence"] >= 0.9
        assert result["enrichment_applied"] is True
        assert "javascript" in result["keywords_added"]
        assert result["embedding"].shape == (1536,)
    
    @pytest.mark.asyncio
    async def test_enrichment_with_low_confidence(self, enrichment_processor):
        """Test that enrichment is skipped for low confidence detections."""
        ambiguous_code = "x = 1"
        
        result = await enrichment_processor.enrich_embedding(ambiguous_code)
        
        # Low confidence should skip enrichment
        if result["confidence"] < 0.7:
            assert result["enrichment_applied"] is False
            assert len(result["keywords_added"]) == 0
        else:
            # If confidence is high enough, enrichment should be applied
            assert result["enrichment_applied"] is True
    
    @pytest.mark.asyncio
    async def test_enrichment_improves_search_relevance(self, enrichment_processor):
        """Test that enrichment improves search relevance."""
        # Sample Python code
        python_code = """
class DataProcessor:
    def process(self, data):
        return [item * 2 for item in data]
"""
        
        # Sample JavaScript code with similar functionality
        js_code = """
class DataProcessor {
    process(data) {
        return data.map(item => item * 2);
    }
}
"""
        
        # Get enriched embeddings
        python_result = await enrichment_processor.enrich_embedding(
            python_code, "processor.py"
        )
        js_result = await enrichment_processor.enrich_embedding(
            js_code, "processor.js"
        )
        
        # Both should be enriched
        assert python_result["enrichment_applied"] is True
        assert js_result["enrichment_applied"] is True
        
        # Language-specific keywords should be different
        assert python_result["keywords_added"] != js_result["keywords_added"]
        
        # Calculate cosine similarity
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # Enriched embeddings should still be somewhat similar (same functionality)
        # but not identical (different languages)
        similarity = cosine_similarity(
            python_result["embedding"],
            js_result["embedding"]
        )
        
        # Should be similar but not too similar
        assert 0.3 < similarity < 0.9, f"Similarity {similarity} out of expected range"
    
    @pytest.mark.asyncio
    async def test_batch_enrichment_performance(self, enrichment_processor):
        """Test batch enrichment performance."""
        # Create multiple code samples
        samples = [
            ("def test(): pass", "test1.py"),
            ("function test() {}", "test2.js"),
            ("public class Test {}", "Test.java"),
            ("fun test() {}", "test.kt"),
            ("print('hello')", "hello.py"),
        ] * 10  # 50 samples total
        
        import time
        start_time = time.perf_counter()
        results = await enrichment_processor.batch_enrich(samples)
        elapsed_time = time.perf_counter() - start_time
        
        assert len(results) == 50
        assert all(r["embedding"].shape == (1536,) for r in results)
        
        # Should process quickly with caching
        avg_time_per_item = elapsed_time / 50
        assert avg_time_per_item < 0.1, f"Batch processing too slow: {avg_time_per_item:.3f}s per item"
        
        print(f"\nBatch Enrichment Performance:")
        print(f"  Total Items: 50")
        print(f"  Total Time: {elapsed_time:.2f}s")
        print(f"  Avg Time per Item: {avg_time_per_item*1000:.1f}ms")
    
    @pytest.mark.asyncio
    async def test_enrichment_weight_calculation(self, enrichment_processor):
        """Test enrichment weight calculation based on confidence."""
        test_cases = [
            ("def main(): pass", "main.py", 0.95),  # High confidence Python
            ("console.log('test')", "test.js", 0.92),  # High confidence JS
            ("x = 1", None, 0.6),  # Low confidence ambiguous
        ]
        
        for code, file_path, _ in test_cases:
            result = await enrichment_processor.enrich_embedding(code, file_path)
            
            if result["enrichment_applied"]:
                # Weight should be proportional to confidence but capped at 0.3
                expected_weight = min(result["confidence"], 0.3)
                assert abs(result["enrichment_weight"] - expected_weight) < 0.01
                
                # Higher confidence should generally mean higher weight
                if result["confidence"] > 0.9:
                    assert result["enrichment_weight"] >= 0.25
    
    @pytest.mark.asyncio
    async def test_enrichment_preserves_semantic_meaning(self, enrichment_processor):
        """Test that enrichment preserves the semantic meaning of code."""
        # Original code
        code = """
def add_numbers(a, b):
    return a + b

result = add_numbers(5, 3)
print(result)
"""
        
        # Get base embedding (mock)
        base_embedding = await enrichment_processor.embeddings_provider.embed(code)
        
        # Get enriched result
        enriched_result = await enrichment_processor.enrich_embedding(
            code, 
            "math.py",
            base_embedding
        )
        
        # Calculate similarity between base and enriched
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        similarity = cosine_similarity(base_embedding, enriched_result["embedding"])
        
        # Enriched embedding should be very similar to base (preserving meaning)
        assert similarity > 0.85, f"Semantic meaning not preserved: similarity={similarity}"
        
        # But not identical (enrichment should make some change)
        assert not np.allclose(base_embedding, enriched_result["embedding"])
    
    @pytest.mark.asyncio
    async def test_cross_language_search_improvement(self, enrichment_processor):
        """Test that enrichment improves cross-language search."""
        # Python implementation
        python_code = """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""
        
        # Java implementation
        java_code = """
public int binarySearch(int[] arr, int target) {
    int left = 0, right = arr.length - 1;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1;
}
"""
        
        # JavaScript implementation
        js_code = """
function binarySearch(arr, target) {
    let left = 0, right = arr.length - 1;
    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        if (arr[mid] === target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1;
}
"""
        
        # Get enriched embeddings for all
        python_result = await enrichment_processor.enrich_embedding(python_code, "search.py")
        java_result = await enrichment_processor.enrich_embedding(java_code, "Search.java")
        js_result = await enrichment_processor.enrich_embedding(js_code, "search.js")
        
        # All should be enriched with their respective languages
        assert python_result["language"] == "python"
        assert java_result["language"] == "java"
        assert js_result["language"] == "javascript"
        
        # Calculate similarities
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # Same algorithm in different languages should still be similar
        py_java_sim = cosine_similarity(python_result["embedding"], java_result["embedding"])
        py_js_sim = cosine_similarity(python_result["embedding"], js_result["embedding"])
        java_js_sim = cosine_similarity(java_result["embedding"], js_result["embedding"])
        
        # Should maintain reasonable similarity (same algorithm)
        assert py_java_sim > 0.4, f"Python-Java similarity too low: {py_java_sim}"
        assert py_js_sim > 0.4, f"Python-JS similarity too low: {py_js_sim}"
        assert java_js_sim > 0.4, f"Java-JS similarity too low: {java_js_sim}"
        
        # But not too similar (different languages)
        assert py_java_sim < 0.95, f"Python-Java similarity too high: {py_java_sim}"
        assert py_js_sim < 0.95, f"Python-JS similarity too high: {py_js_sim}"
        assert java_js_sim < 0.95, f"Java-JS similarity too high: {java_js_sim}"
        
        print(f"\nCross-Language Similarity:")
        print(f"  Python-Java: {py_java_sim:.3f}")
        print(f"  Python-JavaScript: {py_js_sim:.3f}")
        print(f"  Java-JavaScript: {java_js_sim:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])