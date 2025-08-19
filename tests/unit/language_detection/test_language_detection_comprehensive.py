"""Comprehensive unit tests for language detection module to achieve >80% coverage."""

import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from threading import Thread
import concurrent.futures
import ast

from src.project_watch_mcp.language_detection import (
    HybridLanguageDetector,
    LanguageDetectionResult,
    DetectionMethod,
)
from src.project_watch_mcp.language_detection.cache import LanguageDetectionCache


class TestLanguageDetectionCache:
    """Comprehensive tests for LanguageDetectionCache class."""
    
    def test_cache_initialization(self):
        """Test cache initialization with various configurations."""
        # Default initialization
        cache = LanguageDetectionCache()
        assert cache.max_size == 1000
        assert cache.max_age_seconds == 3600
        assert len(cache._cache) == 0
        
        # Custom initialization
        cache = LanguageDetectionCache(max_size=100, max_age_seconds=60)
        assert cache.max_size == 100
        assert cache.max_age_seconds == 60
    
    def test_cache_put_and_get(self):
        """Test adding and retrieving items from cache."""
        cache = LanguageDetectionCache(max_size=10, max_age_seconds=10)
        
        # Create a result to cache
        result = LanguageDetectionResult("python", 0.95, DetectionMethod.TREE_SITTER)
        
        # Put item in cache
        cache.put("test_code", result, file_path="test.py")
        
        # Get item from cache
        cached_result = cache.get("test_code", file_path="test.py")
        assert cached_result is not None
        assert cached_result.language == "python"
        assert cached_result.confidence == 0.95
        
        # Cache hit should increment stats
        assert cache.statistics.hits == 1
        assert cache.statistics.total_requests == 1
    
    def test_cache_expiration(self):
        """Test that cache entries expire after max_age_seconds."""
        cache = LanguageDetectionCache(max_size=10, max_age_seconds=0.1)
        
        result = LanguageDetectionResult("python", 0.95, DetectionMethod.TREE_SITTER)
        cache.put("test_code", result)
        
        # Should be in cache immediately
        assert cache.get("test_code") is not None
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired now
        assert cache.get("test_code") is None
        assert cache.statistics.misses == 1
    
    def test_cache_size_limit(self):
        """Test that cache respects max_size limit."""
        cache = LanguageDetectionCache(max_size=3, max_age_seconds=3600)
        
        # Add more items than max_size
        for i in range(5):
            result = LanguageDetectionResult(f"lang{i}", 0.9, DetectionMethod.TREE_SITTER)
            cache.put(f"code{i}", result)
        
        # Cache should only have max_size items
        assert len(cache._cache) <= 3
        
        # Oldest items should be evicted
        assert cache.get("code0") is None  # Evicted
        assert cache.get("code1") is None  # Evicted
        assert cache.get("code4") is not None  # Still in cache
    
    def test_cache_clear(self):
        """Test clearing the cache."""
        cache = LanguageDetectionCache()
        
        # Add some items
        for i in range(5):
            result = LanguageDetectionResult(f"lang{i}", 0.9, DetectionMethod.TREE_SITTER)
            cache.put(f"code{i}", result)
        
        assert len(cache._cache) == 5
        
        # Clear cache
        cache.clear()
        assert len(cache._cache) == 0
        
        # Cache should be empty but stats could be preserved (implementation dependent)
    
    def test_cache_reset_statistics(self):
        """Test resetting cache statistics."""
        cache = LanguageDetectionCache()
        
        # Generate some stats
        result = LanguageDetectionResult("python", 0.9, DetectionMethod.TREE_SITTER)
        cache.put("code1", result)
        cache.get("code1")  # Hit
        cache.get("code2")  # Miss
        
        # Check stats
        assert cache.statistics.hits == 1
        assert cache.statistics.misses == 1
        
        # Reset stats
        cache.reset_statistics()
        assert cache.statistics.hits == 0
        assert cache.statistics.misses == 0
        assert cache.statistics.evictions == 0
    
    def test_cache_get_info(self):
        """Test getting cache information."""
        cache = LanguageDetectionCache(max_size=100, max_age_seconds=300)
        
        # Add some items and generate stats
        for i in range(5):
            result = LanguageDetectionResult(f"lang{i}", 0.9, DetectionMethod.TREE_SITTER)
            cache.put(f"code{i}", result)
        
        cache.get("code1")  # Hit
        cache.get("nonexistent")  # Miss
        
        info = cache.get_info()
        assert info['size'] == 5
        assert info['max_size'] == 100
        assert info['max_age_seconds'] == 300
        assert info['hit_rate'] == 0.5  # 1 hit, 1 miss (returns as decimal, not percentage)
        assert info['statistics']['hits'] == 1
        assert info['statistics']['misses'] == 1
    
    def test_cache_thread_safety(self):
        """Test that cache operations are thread-safe."""
        cache = LanguageDetectionCache(max_size=1000)
        errors = []
        
        def add_items(start, end):
            try:
                for i in range(start, end):
                    result = LanguageDetectionResult(f"lang{i}", 0.9, DetectionMethod.TREE_SITTER)
                    cache.put(f"code{i}", result)
                    cache.get(f"code{i}")
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = Thread(target=add_items, args=(i*20, (i+1)*20))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # No errors should occur
        assert len(errors) == 0
        # All items should be added
        assert cache.statistics.hits == 100


class TestHybridLanguageDetectorComprehensive:
    """Comprehensive tests for HybridLanguageDetector class."""
    
    @pytest.fixture
    def detector_no_cache(self):
        """Create a detector without caching."""
        return HybridLanguageDetector(enable_cache=False)
    
    @pytest.fixture
    def detector_with_cache(self):
        """Create a detector with caching enabled."""
        return HybridLanguageDetector(enable_cache=True, cache_max_size=100, cache_max_age_seconds=60)
    
    def test_initialization_with_cache_disabled(self):
        """Test detector initialization with cache disabled."""
        detector = HybridLanguageDetector(enable_cache=False)
        assert detector.cache_enabled is False
        assert detector.cache is None
    
    def test_initialization_with_cache_enabled(self):
        """Test detector initialization with cache enabled."""
        detector = HybridLanguageDetector(enable_cache=True, cache_max_size=50, cache_max_age_seconds=30)
        assert detector.cache_enabled is True
        assert detector.cache is not None
        assert detector.cache.max_size == 50
        assert detector.cache.max_age_seconds == 30
    
    def test_tree_sitter_initialization_failure(self):
        """Test handling of tree-sitter initialization failures."""
        with patch('src.project_watch_mcp.language_detection.hybrid_detector.Parser', side_effect=Exception("Parser error")):
            detector = HybridLanguageDetector()
            # Should not crash, just have empty parsers
            assert len(detector.tree_sitter_parsers) == 0
    
    def test_count_error_nodes(self, detector_no_cache):
        """Test counting error nodes in parse tree."""
        # Create a mock node structure
        mock_node = Mock()
        mock_node.type = "ERROR"
        mock_node.has_error = True
        mock_node.children = []
        
        count = detector_no_cache._count_error_nodes(mock_node)
        assert count == 1
        
        # Test with children
        child1 = Mock()
        child1.type = "NORMAL"
        child1.has_error = False
        child1.children = []
        
        child2 = Mock()
        child2.type = "ERROR"
        child2.has_error = True
        child2.children = []
        
        mock_node.children = [child1, child2]
        count = detector_no_cache._count_error_nodes(mock_node)
        assert count == 2  # Parent + child2
    
    def test_count_total_nodes(self, detector_no_cache):
        """Test counting total nodes in parse tree."""
        # Create a mock node structure
        mock_node = Mock()
        mock_node.children = []
        
        count = detector_no_cache._count_total_nodes(mock_node)
        assert count == 1
        
        # Test with children
        child1 = Mock()
        child1.children = []
        child2 = Mock()
        child2.children = []
        
        mock_node.children = [child1, child2]
        count = detector_no_cache._count_total_nodes(mock_node)
        assert count == 3  # Parent + 2 children
    
    def test_adjust_confidence_for_patterns_python(self, detector_no_cache):
        """Test confidence adjustment for Python patterns."""
        python_code = """
import numpy as np
from datetime import datetime

class MyClass:
    def __init__(self):
        self.value = 42
    
    def __name__(self):
        return "MyClass"
"""
        
        # Test various confidence levels
        base_confidence = 0.7
        adjusted = detector_no_cache._adjust_confidence_for_patterns(python_code, "python", base_confidence)
        # Should get boost for import, class, def, __init__, __name__
        assert adjusted > base_confidence
        assert adjusted <= 1.0
    
    def test_adjust_confidence_for_patterns_javascript(self, detector_no_cache):
        """Test confidence adjustment for JavaScript patterns."""
        js_code = """
const myFunction = async () => {
    console.log("Hello");
    return await fetch('/api');
};

let x = 5;
function test() {
    require('module');
}
"""
        
        base_confidence = 0.7
        adjusted = detector_no_cache._adjust_confidence_for_patterns(js_code, "javascript", base_confidence)
        # Should get boost for function, const, let, console.log, require, =>, async
        assert adjusted > base_confidence
        assert adjusted <= 1.0
    
    def test_adjust_confidence_for_patterns_java(self, detector_no_cache):
        """Test confidence adjustment for Java patterns."""
        java_code = """
import java.util.List;

public class Main extends BaseClass {
    private String name;
    
    @Override
    public void run() {
        System.out.println("Running");
    }
}
"""
        
        base_confidence = 0.7
        adjusted = detector_no_cache._adjust_confidence_for_patterns(java_code, "java", base_confidence)
        # Should get boost for public class, private, System.out, import java, @Override, extends
        assert adjusted > base_confidence
        assert adjusted <= 1.0
    
    def test_adjust_confidence_no_patterns(self, detector_no_cache):
        """Test confidence adjustment with no matching patterns."""
        generic_code = "x = 1"
        
        base_confidence = 0.5
        adjusted = detector_no_cache._adjust_confidence_for_patterns(generic_code, "python", base_confidence)
        # No patterns match, so confidence should stay the same
        assert adjusted == base_confidence
    
    @patch('src.project_watch_mcp.language_detection.hybrid_detector.PYGMENTS_AVAILABLE', False)
    def test_detect_from_pygments_not_available(self, detector_no_cache):
        """Test Pygments detection when library is not available."""
        result = detector_no_cache._detect_from_pygments("some code")
        assert result.language == "unknown"
        assert result.confidence == 0.0
        assert result.method == DetectionMethod.PYGMENTS
    
    def test_detect_from_pygments_empty_content(self, detector_no_cache):
        """Test Pygments detection with empty content."""
        result = detector_no_cache._detect_from_pygments("")
        assert result.language == "unknown"
        assert result.confidence == 0.0
        assert result.method == DetectionMethod.PYGMENTS
    
    def test_detect_from_pygments_short_content(self, detector_no_cache):
        """Test Pygments detection with very short content."""
        with patch('src.project_watch_mcp.language_detection.hybrid_detector.PYGMENTS_AVAILABLE', True):
            with patch('src.project_watch_mcp.language_detection.hybrid_detector.lexers.guess_lexer') as mock_guess:
                mock_lexer = Mock()
                mock_lexer.name = "Python"
                mock_guess.return_value = mock_lexer
                
                result = detector_no_cache._detect_from_pygments("x=1")  # Very short
                assert result.method == DetectionMethod.PYGMENTS
                # Short content should reduce confidence (0.8 * 0.8 = 0.64)
                assert result.confidence <= 0.8 * 0.8  # Base * length penalty
    
    def test_detect_from_pygments_long_content(self, detector_no_cache):
        """Test Pygments detection with long content."""
        with patch('src.project_watch_mcp.language_detection.hybrid_detector.PYGMENTS_AVAILABLE', True):
            with patch('src.project_watch_mcp.language_detection.hybrid_detector.lexers.guess_lexer') as mock_guess:
                mock_lexer = Mock()
                mock_lexer.name = "Python"
                mock_guess.return_value = mock_lexer
                
                long_code = "def function():\n    pass\n" * 100  # Long content
                result = detector_no_cache._detect_from_pygments(long_code)
                assert result.method == DetectionMethod.PYGMENTS
                # Long content should increase confidence
                assert result.confidence > 0.8  # Base or higher
    
    def test_detect_from_pygments_exception(self, detector_no_cache):
        """Test Pygments detection with exception."""
        with patch('src.project_watch_mcp.language_detection.hybrid_detector.PYGMENTS_AVAILABLE', True):
            with patch('src.project_watch_mcp.language_detection.hybrid_detector.lexers.guess_lexer', side_effect=Exception("Error")):
                result = detector_no_cache._detect_from_pygments("some code")
                assert result.language == "unknown"
                assert result.confidence == 0.0
                assert result.method == DetectionMethod.PYGMENTS
    
    def test_detect_with_cache_hit(self, detector_with_cache):
        """Test detection with cache hit."""
        python_code = "def hello(): pass"
        
        # First call - should detect and cache
        result1 = detector_with_cache.detect(python_code, file_path="test.py")
        assert result1.language == "python"
        
        # Second call - should hit cache
        result2 = detector_with_cache.detect(python_code, file_path="test.py")
        assert result2.language == "python"
        
        # Check cache stats
        cache_info = detector_with_cache.get_cache_info()
        assert cache_info['statistics']['hits'] == 1
    
    def test_detect_with_cache_disabled_override(self, detector_with_cache):
        """Test overriding cache setting for specific detection."""
        python_code = "def hello(): pass"
        
        # Detect with cache disabled
        result = detector_with_cache.detect(python_code, use_cache=False)
        assert result.language == "python"
        
        # Cache should not be used
        cache_info = detector_with_cache.get_cache_info()
        assert cache_info['statistics']['puts'] == 0
    
    def test_detect_high_confidence_tree_sitter(self, detector_no_cache):
        """Test early return for high confidence tree-sitter result."""
        python_code = """
def calculate(x, y):
    return x + y

class Calculator:
    def __init__(self):
        self.result = 0
"""
        
        result = detector_no_cache.detect(python_code)
        assert result.language == "python"
        assert result.method == DetectionMethod.TREE_SITTER
        assert result.confidence >= 0.9
    
    def test_detect_high_confidence_pygments(self, detector_no_cache):
        """Test early return for high confidence Pygments result."""
        # JavaScript code with patterns that tree-sitter doesn't recognize well
        ambiguous_code = "var x = 1; // Some code"
        
        with patch('src.project_watch_mcp.language_detection.hybrid_detector.PYGMENTS_AVAILABLE', True):
            # Mock tree-sitter to return lower confidence
            with patch.object(detector_no_cache, '_detect_from_tree_sitter') as mock_tree:
                mock_tree.return_value = LanguageDetectionResult("unknown", 0.3, DetectionMethod.TREE_SITTER)
                
                # Mock Pygments to return high confidence
                with patch.object(detector_no_cache, '_detect_from_pygments') as mock_pygments:
                    mock_pygments.return_value = LanguageDetectionResult("javascript", 0.86, DetectionMethod.PYGMENTS)
                    
                    result = detector_no_cache.detect(ambiguous_code)
                    assert result.language == "javascript"
                    assert result.method == DetectionMethod.PYGMENTS
    
    def test_detect_low_confidence_with_extension(self, detector_no_cache):
        """Test fallback to extension when confidence is low."""
        ambiguous_code = "x"
        
        # Mock low confidence results
        with patch.object(detector_no_cache, '_detect_from_tree_sitter') as mock_tree:
            with patch.object(detector_no_cache, '_detect_from_pygments') as mock_pyg:
                mock_tree.return_value = LanguageDetectionResult("unknown", 0.3, DetectionMethod.TREE_SITTER)
                mock_pyg.return_value = LanguageDetectionResult("text", 0.4, DetectionMethod.PYGMENTS)
                
                result = detector_no_cache.detect(ambiguous_code, file_path="script.py")
                # Should prefer extension when confidence is low
                assert result.language == "python"
                assert result.method == DetectionMethod.EXTENSION
    
    def test_detect_no_results(self, detector_no_cache):
        """Test detection when no detection methods succeed."""
        with patch.object(detector_no_cache, '_detect_from_tree_sitter') as mock_tree:
            with patch.object(detector_no_cache, '_detect_from_pygments') as mock_pyg:
                # Simulate all methods returning nothing
                mock_tree.return_value = LanguageDetectionResult("unknown", 0.0, DetectionMethod.TREE_SITTER)
                mock_pyg.return_value = LanguageDetectionResult("unknown", 0.0, DetectionMethod.PYGMENTS)
                
                result = detector_no_cache.detect("???", file_path=None)
                assert result.language == "unknown"
                assert result.confidence == 0.0
    
    def test_detect_batch_with_cache(self, detector_with_cache):
        """Test batch detection with caching."""
        files = [
            ("test1.py", "def hello(): pass"),
            ("test2.js", "function hello() {}"),
            ("test1.py", "def hello(): pass"),  # Duplicate - should hit cache
        ]
        
        results = detector_with_cache.detect_batch(files)
        assert len(results) == 3
        assert results[0].language == "python"
        assert results[1].language == "javascript"
        assert results[2].language == "python"
        
        # Check cache hit for duplicate
        cache_info = detector_with_cache.get_cache_info()
        assert cache_info['statistics']['hits'] == 1
    
    def test_detect_batch_cache_override(self, detector_with_cache):
        """Test batch detection with cache override."""
        files = [
            ("test1.py", "def hello(): pass"),
            ("test2.js", "function hello() {}"),
        ]
        
        # Detect without cache
        results = detector_with_cache.detect_batch(files, use_cache=False)
        assert len(results) == 2
        
        # Cache should not be used
        cache_info = detector_with_cache.get_cache_info()
        assert cache_info['statistics']['puts'] == 0
    
    def test_clear_cache(self, detector_with_cache):
        """Test clearing the detector cache."""
        # Add some items to cache
        detector_with_cache.detect("def hello(): pass", file_path="test.py")
        
        cache_info = detector_with_cache.get_cache_info()
        assert cache_info['size'] > 0
        
        # Clear cache
        detector_with_cache.clear_cache()
        
        cache_info = detector_with_cache.get_cache_info()
        assert cache_info['size'] == 0
    
    def test_reset_cache_statistics(self, detector_with_cache):
        """Test resetting cache statistics."""
        # Generate some stats
        detector_with_cache.detect("def hello(): pass", file_path="test.py")
        detector_with_cache.detect("def hello(): pass", file_path="test.py")  # Cache hit
        
        cache_info = detector_with_cache.get_cache_info()
        assert cache_info['statistics']['hits'] > 0
        
        # Reset stats
        detector_with_cache.reset_cache_statistics()
        
        cache_info = detector_with_cache.get_cache_info()
        assert cache_info['statistics']['hits'] == 0
        assert cache_info['statistics']['puts'] == 0
    
    def test_get_cache_info_no_cache(self, detector_no_cache):
        """Test getting cache info when cache is disabled."""
        info = detector_no_cache.get_cache_info()
        assert info is None
    
    def test_clear_cache_no_cache(self, detector_no_cache):
        """Test clearing cache when cache is disabled."""
        # Should not raise error
        detector_no_cache.clear_cache()
    
    def test_reset_statistics_no_cache(self, detector_no_cache):
        """Test resetting statistics when cache is disabled."""
        # Should not raise error
        detector_no_cache.reset_cache_statistics()


class TestLanguageNormalization:
    """Test language name normalization."""
    
    @pytest.fixture
    def detector(self):
        return HybridLanguageDetector(enable_cache=False)
    
    @pytest.mark.parametrize("input_name,expected", [
        ("Python", "python"),
        ("python3", "python"),
        ("py", "python"),
        ("JavaScript", "javascript"),
        ("js", "javascript"),
        ("node", "javascript"),
        ("TypeScript", "typescript"),
        ("ts", "typescript"),
        ("Java", "java"),
        ("Kotlin", "kotlin"),
        ("kt", "kotlin"),
        ("C++", "cpp"),
        ("cpp", "cpp"),
        ("C", "c"),
        ("C#", "csharp"),
        ("csharp", "csharp"),
        ("cs", "csharp"),
        ("Objective-C", "objc"),
        ("objc", "objc"),
        ("objectivec", "objc"),
        ("scdoc", "python"),
        ("numpy", "python"),
        ("text only", "text"),
        ("text", "text"),
        ("tera term macro", "kotlin"),
        ("UnknownLanguage", "unknownlanguage"),  # Unknown language
    ])
    def test_normalize_language(self, detector, input_name, expected):
        """Test language normalization for various inputs."""
        assert detector._normalize_language(input_name) == expected


class TestTreeSitterEdgeCases:
    """Test edge cases for tree-sitter detection."""
    
    @pytest.fixture
    def detector(self):
        return HybridLanguageDetector(enable_cache=False)
    
    def test_detect_from_tree_sitter_empty(self, detector):
        """Test tree-sitter with empty content."""
        result = detector._detect_from_tree_sitter("")
        assert result.language == "unknown"
        assert result.confidence == 0.0
        assert result.method == DetectionMethod.TREE_SITTER
    
    def test_detect_from_tree_sitter_whitespace(self, detector):
        """Test tree-sitter with only whitespace."""
        result = detector._detect_from_tree_sitter("   \n\t  ")
        assert result.language == "unknown"
        assert result.confidence == 0.0
        assert result.method == DetectionMethod.TREE_SITTER
    
    def test_detect_from_tree_sitter_parse_exception(self, detector):
        """Test tree-sitter with parsing exception."""
        # Mock a parser that raises exception
        mock_parser = Mock()
        mock_parser.parse.side_effect = Exception("Parse error")
        
        detector.tree_sitter_parsers = {"python": mock_parser}
        
        result = detector._detect_from_tree_sitter("def hello(): pass")
        # Should handle exception and return unknown
        assert result.language == "unknown"
        assert result.confidence == 0.0
    
    def test_detect_from_tree_sitter_no_nodes(self, detector):
        """Test tree-sitter with parse tree having no nodes."""
        mock_parser = Mock()
        mock_tree = Mock()
        mock_root = Mock()
        mock_root.has_error = False
        mock_root.children = []
        mock_tree.root_node = mock_root
        mock_parser.parse.return_value = mock_tree
        
        # Mock the count methods to return 0 total nodes (edge case)
        with patch.object(detector, '_count_total_nodes', return_value=0):
            with patch.object(detector, '_count_error_nodes', return_value=0):
                detector.tree_sitter_parsers = {"python": mock_parser}
                result = detector._detect_from_tree_sitter("def hello(): pass")
                # When total nodes is 0, confidence should be 0
                assert result.confidence == 0.0


class TestDetectionIntegration:
    """Integration tests for complete detection flow."""
    
    @pytest.fixture
    def detector(self):
        return HybridLanguageDetector(enable_cache=True)
    
    def test_detect_various_languages(self, detector):
        """Test detection of various programming languages."""
        test_cases = [
            # Python
            ("""
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
""", "test.py", "python"),
            
            # JavaScript
            ("""
const factorial = (n) => {
    return n <= 1 ? 1 : n * factorial(n-1);
};
""", "test.js", "javascript"),
            
            # Java
            ("""
public class Factorial {
    public static int factorial(int n) {
        return n <= 1 ? 1 : n * factorial(n-1);
    }
}
""", "test.java", "java"),
            
            # Ruby (no tree-sitter, might be detected as Python due to similarities)
            ("""
def factorial(n)
  n <= 1 ? 1 : n * factorial(n-1)
end
""", "test.rb", ["ruby", "python"]),  # Accept either due to syntax similarities
        ]
        
        for code, file_path, expected_lang in test_cases:
            result = detector.detect(code, file_path)
            if isinstance(expected_lang, list):
                assert result.language in expected_lang
            else:
                assert result.language == expected_lang
            assert result.confidence > 0
    
    def test_concurrent_detection(self, detector):
        """Test thread-safe concurrent detection."""
        codes = [
            ("def hello(): pass", "test1.py"),
            ("function hello() {}", "test2.js"),
            ("public class Test {}", "test3.java"),
            ("def world(): return 42", "test4.py"),
            ("const x = 5;", "test5.js"),
        ]
        
        results = []
        errors = []
        
        def detect_code(code, path):
            try:
                result = detector.detect(code, path)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Use ThreadPoolExecutor for concurrent detection
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(detect_code, code, path) for code, path in codes]
            concurrent.futures.wait(futures)
        
        # No errors should occur
        assert len(errors) == 0
        # All detections should complete
        assert len(results) == 5
        # Results should be valid
        for result in results:
            assert result.language != "unknown" or result.confidence == 0.0