"""Integration tests for language detection with caching."""

import pytest
import time
from unittest.mock import Mock, patch

from src.project_watch_mcp.language_detection import (
    HybridLanguageDetector,
    LanguageDetectionResult,
    DetectionMethod,
)


class TestLanguageDetectionCaching:
    """Test language detection with caching enabled."""
    
    def test_cache_enabled_by_default(self):
        """Test that cache is enabled by default."""
        detector = HybridLanguageDetector()
        assert detector.cache_enabled is True
        assert detector.cache is not None
        
        # Get cache info
        info = detector.get_cache_info()
        assert info is not None
        assert info["size"] == 0
        assert info["max_size"] == 1000
    
    def test_cache_can_be_disabled(self):
        """Test that cache can be disabled."""
        detector = HybridLanguageDetector(enable_cache=False)
        assert detector.cache_enabled is False
        assert detector.cache is None
        
        # Get cache info should return None
        info = detector.get_cache_info()
        assert info is None
    
    def test_detection_uses_cache(self):
        """Test that detection uses cache for repeated calls."""
        detector = HybridLanguageDetector()
        
        python_code = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
"""
        
        # First detection - should miss cache
        result1 = detector.detect(python_code)
        assert result1.language == "python"
        
        # Check cache statistics
        info = detector.get_cache_info()
        assert info["statistics"]["misses"] == 1
        assert info["statistics"]["hits"] == 0
        
        # Second detection - should hit cache
        result2 = detector.detect(python_code)
        assert result2.language == "python"
        assert result2 == result1  # Should be the same result
        
        # Check cache statistics
        info = detector.get_cache_info()
        assert info["statistics"]["hits"] == 1
        assert info["statistics"]["misses"] == 1
        assert info["hit_rate"] == 0.5
    
    def test_cache_respects_file_path(self):
        """Test that cache considers file path in key."""
        detector = HybridLanguageDetector()
        
        # Ambiguous content that could be multiple languages
        content = "print('Hello')"
        
        # Detect with Python file extension
        result_py = detector.detect(content, "test.py")
        assert result_py.language == "python"
        
        # Detect with Ruby file extension (should not hit cache)
        result_rb = detector.detect(content, "test.rb")
        # Could be detected as ruby or python depending on detection method
        
        # Check that we had two cache misses (different keys)
        info = detector.get_cache_info()
        assert info["statistics"]["misses"] == 2
        
        # Repeat detection with same file paths - should hit cache
        result_py2 = detector.detect(content, "test.py")
        assert result_py2 == result_py
        
        info = detector.get_cache_info()
        assert info["statistics"]["hits"] == 1
    
    def test_cache_can_be_bypassed(self):
        """Test that cache can be bypassed for specific detections."""
        detector = HybridLanguageDetector()
        
        js_code = "console.log('Hello, World!');"
        
        # First detection with cache
        result1 = detector.detect(js_code)
        
        # Second detection bypassing cache
        result2 = detector.detect(js_code, use_cache=False)
        
        # Check statistics - should have only one miss (first call)
        info = detector.get_cache_info()
        assert info["statistics"]["misses"] == 1
        assert info["statistics"]["hits"] == 0
    
    def test_batch_detection_with_cache(self):
        """Test batch detection uses cache effectively."""
        detector = HybridLanguageDetector()
        
        files = [
            ("test1.py", "def hello(): pass"),
            ("test2.js", "function hello() {}"),
            ("test1.py", "def hello(): pass"),  # Duplicate
            ("test3.java", "public class Test {}"),
        ]
        
        results = detector.detect_batch(files)
        
        assert len(results) == 4
        assert results[0].language == "python"
        assert results[1].language == "javascript"
        assert results[2].language == "python"
        assert results[3].language == "java"
        
        # Check cache statistics
        info = detector.get_cache_info()
        # Should have 3 misses (unique files) and 1 hit (duplicate)
        assert info["statistics"]["misses"] == 3
        assert info["statistics"]["hits"] == 1
    
    def test_cache_clear(self):
        """Test clearing the cache."""
        detector = HybridLanguageDetector()
        
        # Add some detections to cache
        detector.detect("def hello(): pass", "test.py")
        detector.detect("function hello() {}", "test.js")
        
        info = detector.get_cache_info()
        assert info["size"] == 2
        
        # Clear cache
        detector.clear_cache()
        
        info = detector.get_cache_info()
        assert info["size"] == 0
        # Statistics should be preserved
        assert info["statistics"]["misses"] == 2
    
    def test_cache_statistics_reset(self):
        """Test resetting cache statistics."""
        detector = HybridLanguageDetector()
        
        # Generate some statistics
        detector.detect("def hello(): pass")
        detector.detect("def hello(): pass")  # Hit
        
        info = detector.get_cache_info()
        assert info["statistics"]["hits"] == 1
        assert info["statistics"]["misses"] == 1
        
        # Reset statistics
        detector.reset_cache_statistics()
        
        info = detector.get_cache_info()
        assert info["statistics"]["hits"] == 0
        assert info["statistics"]["misses"] == 0
        assert info["size"] == 1  # Cache content preserved
    
    def test_cache_improves_performance(self):
        """Test that cache improves performance for repeated detections."""
        detector = HybridLanguageDetector()
        
        # Complex Python code that takes time to analyze
        complex_code = """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class MLPipeline:
    def __init__(self, n_estimators=100):
        self.model = RandomForestClassifier(n_estimators=n_estimators)
        self.data = None
        self.features = None
        self.target = None
    
    def load_data(self, filepath):
        self.data = pd.read_csv(filepath)
        return self
    
    def preprocess(self):
        # Remove missing values
        self.data = self.data.dropna()
        
        # Feature engineering
        self.features = self.data.drop('target', axis=1)
        self.target = self.data['target']
        
        return self
    
    def train(self, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=test_size
        )
        
        self.model.fit(X_train, y_train)
        score = self.model.score(X_test, y_test)
        
        return score
    
    def predict(self, X):
        return self.model.predict(X)
"""
        
        # Measure time for first detection (cache miss)
        start_time = time.perf_counter()
        result1 = detector.detect(complex_code)
        first_detection_time = time.perf_counter() - start_time
        
        # Measure time for second detection (cache hit)
        start_time = time.perf_counter()
        result2 = detector.detect(complex_code)
        cached_detection_time = time.perf_counter() - start_time
        
        # Cache should be significantly faster
        assert result1 == result2
        assert cached_detection_time < first_detection_time
        
        # Verify cache was used
        info = detector.get_cache_info()
        assert info["statistics"]["hits"] == 1
        assert info["statistics"]["misses"] == 1
    
    def test_cache_with_custom_settings(self):
        """Test cache with custom size and age settings."""
        detector = HybridLanguageDetector(
            enable_cache=True,
            cache_max_size=5,
            cache_max_age_seconds=1
        )
        
        # Fill cache to max size
        for i in range(5):
            detector.detect(f"content_{i} = {i}")
        
        info = detector.get_cache_info()
        assert info["size"] == 5
        assert info["max_size"] == 5
        
        # Add one more - should evict oldest
        detector.detect("content_new = 'new'")
        
        info = detector.get_cache_info()
        assert info["size"] == 5  # Still at max
        assert info["statistics"]["evictions"] == 1
        
        # Test expiration
        detector.detect("expire_test = True")
        time.sleep(1.1)  # Wait for expiration
        
        # Try to get expired entry
        detector.detect("expire_test = True")  # Should miss due to expiration
        
        info = detector.get_cache_info()
        # Should have two misses for "expire_test"
        assert info["statistics"]["misses"] >= 2
    
    @pytest.mark.parametrize("language,code", [
        ("python", "def main():\n    print('Hello')"),
        ("javascript", "const hello = () => console.log('Hello');"),
        ("java", "public class Main { public static void main(String[] args) {} }"),
    ])
    def test_cache_with_different_languages(self, language, code):
        """Test that cache works correctly for different languages."""
        detector = HybridLanguageDetector()
        
        # First detection
        result1 = detector.detect(code)
        assert result1.language == language
        
        # Second detection - should use cache
        result2 = detector.detect(code)
        assert result2 == result1
        
        # Verify cache hit
        info = detector.get_cache_info()
        assert info["statistics"]["hits"] >= 1